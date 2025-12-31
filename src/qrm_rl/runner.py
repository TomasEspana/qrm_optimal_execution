import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
from numba import njit
from torch import nn
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from qrm_rl.callbacks import InfoLoggerCallback
import shap
import pandas as pd
import os

from qrm_rl.agents.benchmark_strategies import TWAPAgent, POPVAgent
from qrm_core.intensity import IntensityTable


"""
    GENERAL DESCRIPTION:
        Runner class to manage training and testing of RL agents in the QRM environment.
"""


class RLRunner:
    def __init__(self, config, load_model_path=None):

        # Unpack config
        self.cfg = config
        self.agent_type = config['agent_type'].lower()
        self.mode = config['mode']
        self.test_mode = (self.mode == 'test')
        self.episodes = config['episodes']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = None
        self.load_model_path = load_model_path
        self.model = None

        # Seeds
        np.random.seed(config['seed'])
        @njit
        def _init_numba(seed): np.random.seed(seed)
        _init_numba(config['seed'])

        # WandB init
        if self.cfg['mode'] == 'train':
            wandb.init(
                project="QRM_RL_Agent",
                name=f"{self.mode}",
                config=config, 
                sync_tensorboard=False,
            )

        # Intensity table
        inten_arr = np.load(config['folder_path_intensity_table'] + config['file_name'])
        K, Q1, *_ = inten_arr.shape
        inten_table = IntensityTable(max_depth=K, max_queue=Q1-1)
        inten_table._data = inten_arr

        # Environment
        self.env = gym.make(
            "QRM-v0",
            intensity_table=inten_table,
            actions=config['actions'],
            theta=config['theta'],
            theta_reinit=config['theta_reinit'],
            tick=config['tick'],
            arrival_price=config['arrival_price'],
            inv_bid_file=config['folder_path_invariant'] + config['file_name_bid'],
            inv_ask_file=config['folder_path_invariant'] + config['file_name_ask'],
            trader_times=config['trader_times'],
            initial_inventory=config['initial_inventory'],
            time_horizon=config['time_horizon'],
            final_penalty=config['final_penalty'],
            risk_aversion=config['risk_aversion'],
            price_offset=config['price_offset'],
            price_std=config['price_std'],
            vol_offset=config['vol_offset'],
            vol_std=config['vol_std'],
            max_events=config['max_events'],
            max_events_intra=config['max_events_intra'],
            history_size=config['history_size'],
            state_dim=config['state_dim'],
            action_dim=config['action_dim'], 
            aes=np.array(config['aes']), 
            test_mode=self.test_mode
        )

        self.env = Monitor(self.env)
        self.agent_name_map = {
            DQN: 'ddqn',
            TWAPAgent: 'twap', 
            POPVAgent: 'popv'
        }


    # Build DQN agent
    def _build_dqn(self):
        if self.model is not None:
            return

        # NN architecture
        policy_kwargs = dict(
            net_arch=[30, 30, 30, 30, 30], 
            activation_fn=nn.LeakyReLU,
            )
        
        self.model = DQN(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=self.cfg["learning_rate"],
            buffer_size=self.cfg["buffer_size"],
            batch_size=self.cfg["batch_size"],
            learning_starts=self.cfg["learning_starts"],
            gamma=self.cfg["gamma"],
            target_update_interval=self.cfg["target_update_interval"],
            train_freq=self.cfg['train_freq'],
            gradient_steps=self.cfg["gradient_steps"],
            exploration_initial_eps=self.cfg["exploration_initial_eps"],
            exploration_final_eps=self.cfg["exploration_final_eps"],
            exploration_fraction=self.cfg["exploration_fraction"],
            verbose=0,
            policy_kwargs=policy_kwargs,
            device=self.device,
            n_steps=self.cfg["n_steps"]
        )
        self.agent = self.model


    def unwrap_env(self, env):
        """
            Helper function to unwrap nested Gym wrappers to access the underlying base environment.
        """
        while hasattr(env, "env"):
            env = env.env
        return env


    def run(self, agent_info=None):

        print('RUNNING ON DEVICE:', self.device)
        if self.device == 'cpu':
            print("WARNING: Default device is GPU. Check CUDA availability.")
        
            # ===== TRAIN MODE =====
        if self.mode == 'train':

            if self.agent_type == 'ddqn':
                self._build_dqn()
            else:
                raise ValueError(f"Unknown agent type {self.agent_type} for training.")
            
            self.run_id = wandb.run.id
            wandb.run.name = f"{self.agent_type}_{self.run_id}"
            total_steps = self.cfg["total_timesteps"]

            callback = CallbackList([
                  WandbCallback(verbose=2,),
                  InfoLoggerCallback(self.cfg["action_dim"])
                 ])

            self.model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)
            self.model.save(f"./save_model/{self.agent_type}_{self.run_id}.zip")
            wandb.finish()

            if self.agent_type == 'ddqn':

                ### Feature importance
                ## a) SHAP values
                base_output_dir = "./shap_plots"
                os.makedirs(base_output_dir, exist_ok=True)
                output_dir = os.path.join(base_output_dir, self.run_id)
                os.makedirs(output_dir, exist_ok=True)

                sample_size = 200 
                background_size = 500
                buffer = self.model.replay_buffer
                end_idx = buffer.size()
                obs = buffer.observations[:end_idx]
                sample_start = max(0, end_idx - sample_size)
                sample_states = obs[sample_start:end_idx]
                sample_states = sample_states.reshape(sample_states.shape[0], sample_states.shape[2])
                bg_start = max(0, sample_start - background_size)
                background = obs[bg_start:sample_start]
                background = background.reshape(background.shape[0], background.shape[2])

                save_path = os.path.join(output_dir, "shap_data.npz")
                np.savez_compressed(save_path, sample_states=sample_states, background=background)
                print(f"Saved SHAP data to {save_path}")

                # Q-network prediction function
                q_net = self.model.policy.q_net
                def model_predict(x):
                    with torch.no_grad():
                        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
                        return q_net(x_t).cpu().numpy()

                # Compute SHAP values with KernelExplainer
                explainer = shap.KernelExplainer(model_predict, background)
                shap_values = explainer.shap_values(sample_states) # shape (sample_size, num_features, num_actions)

                ## b) Input-gradient analysis
                states_t = torch.tensor(background, dtype=torch.float32, device=self.device, requires_grad=True)
                num_actions = self.cfg['action_dim']
                feature_names = ["inventory", "time", "ask price", "ask volume", "bid volume"]
                feature_names = feature_names[:self.cfg['state_dim']]
                
                gradient_importances = []
                for action_idx in range(num_actions):
                    q_vals = q_net(states_t)[:, action_idx].sum()
                    grads = torch.autograd.grad(q_vals, states_t, retain_graph=True)[0]  # shape (N, num_features)
                    gradient_importances.append(np.abs(grads.detach().cpu().numpy()).mean(axis=0))

                gradient_importances = np.array(gradient_importances)  # shape (num_actions, num_features)


                for action_idx in range(shap_values.shape[2]):
                    values = shap_values[:, :, action_idx]
                    shap_importance = np.abs(values).mean(0)
                    grad_importance = gradient_importances[action_idx]

                    df_compare = pd.DataFrame({
                        "feature": feature_names,
                        "SHAP": shap_importance,
                        "Gradient": grad_importance
                    }).sort_values("SHAP", ascending=False)

                    print(f"\n=== Action {action_idx} Feature Importance ===")
                    print(df_compare)

                    # SHAP summary plot
                    fig = plt.figure()
                    shap.summary_plot(values, sample_states, feature_names=feature_names, 
                                    title=f"SHAP Values for Action {action_idx}", show=False)
                    fig.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"summary_action_{action_idx}.pdf"), bbox_inches='tight')
                    plt.close(fig)

                    fig = plt.figure()
                    shap_values_obj = shap.Explanation(
                        values=values,
                        data=sample_states,
                        feature_names=feature_names
                    )
                    shap.plots.bar(shap_values_obj, show=False)
                    fig.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"bar_action_{action_idx}.pdf"), bbox_inches='tight')
                    plt.close(fig)

                    for feat in shap_values_obj.feature_names:
                        fig = plt.figure()
                        shap.plots.scatter(shap_values_obj[:, feat], color=shap_values_obj, show=False)
                        fig.tight_layout() 
                        plt.savefig(os.path.join(output_dir, f"scatter_action_{action_idx}_{feat}.pdf"), bbox_inches='tight')
                        plt.close(fig)

                    # Gradient bar plot
                    plt.figure()
                    plt.bar(feature_names, grad_importance)
                    plt.title(f"Gradient-based Feature Importance (Action {action_idx})")
                    plt.ylabel("Mean |∂Q/∂s|")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"grad_bar_action_{action_idx}.pdf"), bbox_inches='tight')
                    plt.close()

                self.env.close()
            return
        
            # ===== TEST MODE =====
        else:

            if agent_info in ['DQN']: # RL agent
                self._build_dqn()
                if self.load_model_path is not None:
                    self.model = DQN.load(self.load_model_path, env=self.env, device=self.device)
                else:
                    raise ValueError("In test mode with an RL agent, load_model_path must be provided.")
                self.agent = self.model
            
            # Logging
            mid_prices, mid_prices_events, lob_dataframe, actions_taken, executed_dic, index_actions, ba_vol_dic, bb_vol_dic = {}, {}, {}, {}, {}, {}, {}, {}
            final_is = []

            for ep in range(self.episodes):

                actions, executed, ba_vol, bb_vol = [], [], [], []
                done, ep_reward = False, 0. 

                obs, _ = self.env.reset()
                idx_actions = [self.unwrap_env(self.env)._env.simulator.step]
                k = 1

                while not done:

                    if isinstance(self.agent, (DQN)):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, _, info = self.env.step(action)

                    else:
                        self.agent.k = k
                        if isinstance(self.agent, TWAPAgent):
                            self.unwrap_env(self.env)._twap_execution = True
                            action = self.agent.select_action()
                            obs, reward, done, _, info = self.env.step(action)
                        else:
                            action = self.agent.select_action()
                            obs, reward, done, _, info = self.env.step(action)

                    actions.append(action)    
                    executed.append(info["executed"])
                    ba_vol.append(info["best_ask_volume"])
                    bb_vol.append(info["best_bid_volume"])
                    idx_actions.append(self.unwrap_env(self.env)._env.simulator.step)
                    ep_reward += reward
                    k += 1

                # end-of-episode book-keeping
                unwrapped_env = self.unwrap_env(self.env)
                final_is.append(unwrapped_env._env.final_is)
                actions_taken[ep] = actions
                executed_dic[ep] = executed
                ba_vol_dic[ep] = ba_vol
                bb_vol_dic[ep] = bb_vol
                index_actions[ep] = idx_actions[:-1]
                mid_prices_events[ep] = unwrapped_env._env.simulator.p_mids[:unwrapped_env._env.simulator.step][idx_actions[:-1]].astype(np.float32)

                if not self.cfg['test_save_memory']:
                    lob = unwrapped_env._env.simulator.to_dataframe()
                    lob['time'] = lob['time'].astype(np.float32)
                    lob['p_mid'] = lob['p_mid'].astype(np.float32)
                    lob['p_ref'] = lob['p_ref'].astype(np.float32)
                    lob_dataframe[ep] = lob
                    mid_prices[ep] = unwrapped_env._env.simulator.p_mids[:unwrapped_env._env.simulator.step].astype(np.float32)
                # NB: the mid price observed at index_action corresponds to the price after the action was taken.
                # When plotting, you may want to shift(-1) the index actions to better grasp the change in mid price after the action.

                if self.cfg['logging'] and (ep % self.cfg['logging_every'] == 0):
                    print(f"[{self.mode.upper()}][{ep}/{self.episodes}]  Reward={ep_reward:.2f}")

            dic = {
                "final_is": final_is,
                "lob": lob_dataframe,
                "mid_prices": mid_prices,
                "mid_prices_events": mid_prices_events,
                "actions": actions_taken,
                "executed": executed_dic,
                "index_actions": index_actions, 
                "ba_vol": ba_vol_dic, 
                "bb_vol": bb_vol_dic
            }

            self.env.close()

            return dic, 0