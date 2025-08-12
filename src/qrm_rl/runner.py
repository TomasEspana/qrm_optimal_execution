import matplotlib
matplotlib.use("Agg") 
import wandb
import numpy as np
import torch
from numba import njit
from qrm_rl.agents.benchmark_strategies import TWAPAgent, BackLoadAgent, FrontLoadAgent, RandomAgent, BimodalAgent, BestVolumeAgent
from qrm_core.intensity import IntensityTable
from torch import nn

import qrm_rl.gym_env 
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from qrm_rl.callbacks import InfoLoggerCallback, InjectEpsCallback
from qrm_rl.custom_eps_policy import CustomEpsMlpPolicy
import shap
import pandas as pd
import os
import matplotlib.pyplot as plt

# from contextlib import nullcontext
# from qrm_rl.agents.ddqn import DDQNAgent
# from .market_environment import MarketEnvironment
# from .utils import load_model, save_model


# exploration_mode_dic = {'rl': int(0), 'front_load': int(1), 'back_load': int(2), 'twap': int(3)}


# class CustomDQN(DQN):
#     def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
#         """
#         Override epsilon-greedy exploration to use custom probabilities.
#         """
#         # Exploration phase
#         print('exploration_rate:', self.exploration_rate)
#         if self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate:
#             # Custom probability distribution
#             actions = np.random.choice([0, 1], size=n_envs, p=[0.9, 0.1])
#             # SB3 expects both actions and buffer_actions
#             return actions, actions
#         else:
#             # Default exploitation from parent
#             return super()._sample_action(learning_starts, action_noise, n_envs)

class RLRunner:
    def __init__(self, config, load_model_path=None):
        # Unpack config
        self.cfg = config
        self.mode = config['mode']
        self.test_mode = (self.mode == 'test')
        self.episodes = config['episodes']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prop_greedy_eps = config['exploration_fraction']
        self.agent = None
        self.load_model_path = load_model_path
        self.agent_name_map = {
            DQN: 'ddqn', # CustomDQN: 'ddqn',
            PPO: 'ppo',
            TWAPAgent: 'twap', 
            BackLoadAgent: 'back_load',
            FrontLoadAgent: 'front_load', 
            RandomAgent: 'random', 
            BestVolumeAgent: 'best_volume'
            # InactiveAgent: 'inactive',
            # PassiveAgent: 'passive',
        }

        # Seeds
        np.random.seed(config['seed'])
        @njit
        def _init_numba(seed): np.random.seed(seed)
        _init_numba(config['seed'])

        # WandB init
        wandb.init(
            project="QRM_RL_Agent",
            name=f"{self.mode}",
            config=config, 
            sync_tensorboard=True,
        )

        # Build intensity table
        inten_arr = np.load(config['folder_path_intensity_table'] + config['file_name'])
        K, Qp1, *_ = inten_arr.shape
        inten_table = IntensityTable(max_depth=K, max_queue=Qp1-1)
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
            alpha_ramp=config['alpha_ramp'],
            basic_state=config['basic_state'],
            state_dim=config['state_dim'],
            action_dim=config['action_dim'], 
            aes=np.array(config['aes']), 
            test_mode=self.test_mode
        )

        # Agent
        # Wrap in Monitor for SB3 episode logging
        self.env = Monitor(self.env)

        # SB3 DQN model
        policy_kwargs = dict(
            net_arch=[30, 30, 30, 30, 30], 
            activation_fn=nn.LeakyReLU,
            )
        
        self.model = DQN(
            policy='MlpPolicy', # CustomEpsMlpPolicy
            env=self.env,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            learning_starts=config["learning_starts"],
            gamma=config["gamma"],
            target_update_interval=config["target_update_interval"],
            train_freq=config['train_freq'],
            gradient_steps=config["gradient_steps"],
            exploration_initial_eps=config["exploration_initial_eps"],
            exploration_final_eps=config["exploration_final_eps"],
            exploration_fraction=config["exploration_fraction"],
            verbose=2,
            policy_kwargs=policy_kwargs,
            device=self.device,
            n_steps=config["n_steps"]
        )
        self.agent = self.model

    def unwrap_env(self, env):
        """
        Recursively unwrap Gym wrappers to reach your custom QRMEnv.
        """
        while hasattr(env, "env"):
            env = env.env
        return env

    def run(self):
        agent_type = self.agent_name_map.get(type(self.agent), 'Unknown')
        self.run_id = wandb.run.id
        train_mode = (self.mode == 'train')
        
        if train_mode:
            # ===== TRAIN MODE =====
            wandb.run.name = f"{agent_type}_{self.run_id}"
            total_steps = self.cfg["total_timesteps"]

            callback = CallbackList([
                  WandbCallback(verbose=2,
                                #sync_tensorboard=True,
                                ),
                InfoLoggerCallback(self.cfg["action_dim"]) # InjectEpsCallback()
            ])

            self.model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)
            self.model.save(f"save_model/{agent_type}_{self.run_id}.zip")
            wandb.finish()

            ### Feature importance
            base_output_dir = "shap_plots"
            os.makedirs(base_output_dir, exist_ok=True)
            output_dir = os.path.join(base_output_dir, self.run_id)
            os.makedirs(output_dir, exist_ok=True)

            sample_size = 200 # 200 
            background_size = 500 # 500
            buffer = self.model.replay_buffer
            end_idx = buffer.size()
            obs = buffer.observations[:end_idx]
            sample_start = max(0, end_idx - sample_size)
            bg_start = max(0, sample_start - background_size)
            sample_states = obs[sample_start:end_idx]
            sample_states = sample_states.reshape(sample_states.shape[0], sample_states.shape[2])
            background = obs[bg_start:sample_start]
            background = background.reshape(background.shape[0], background.shape[2])

            save_path = os.path.join(output_dir, "shap_data.npz")
            np.savez_compressed(save_path, sample_states=sample_states, background=background)
            print(f"Saved SHAP data to {save_path}")

            # Q-network prediction function
            q_net = self.model.policy.q_net

            # Wrapper
            def model_predict(x):
                with torch.no_grad():
                    x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
                    return q_net(x_t).cpu().numpy()

            # KernelExplainer
            explainer = shap.KernelExplainer(model_predict, background)

            # Compute SHAP values (limit to 100 states for speed)
            shap_values = explainer.shap_values(sample_states)
            print("shap_values shape:", np.array(shap_values).shape)

            # Gradient feature importance
            states_t = torch.tensor(background, dtype=torch.float32, device=self.device, requires_grad=True)
            num_actions = self.cfg['action_dim']
            feature_names = ["inventory", "time", "ask price", "ask volume", "bid volume"]
            feature_names = feature_names[:num_actions]
            
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

            return
        
        else:
            # ===== TEST MODE =====
            wandb.run.name = f"{agent_type}_test_{self.run_id}"

            # Load SB3 model
            if self.load_model_path is not None:
                self.model = DQN.load(self.load_model_path, env=self.env, device=self.device)
            # Logging
            mid_prices, mid_prices_events, lob_dataframe, actions_taken, executed_dic, index_actions, ba_vol_dic, bb_vol_dic = {}, {}, {}, {}, {}, {}, {}, {}
            final_is = []

            for ep in range(self.episodes):

                actions, executed, ba_vol, bb_vol = [], [], [], []
                done, ep_reward = False, 0. 

                obs, _ = self.env.reset()
                idx_actions = [self.unwrap_env(self.env)._env.simulator.step]  # first index
                k = 1

                while not done:

                    if isinstance(self.agent, (DQN, PPO)):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, _, info = self.env.step(action)

                    else:

                        self.agent.k = k
                        if isinstance(self.agent, TWAPAgent):
                            self.unwrap_env(self.env)._executed = True # quantity, not index
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

                # end-of-episode book-keeping (single ep test; extend if multiple desired)
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
                # NOTE: that the mid price observed at index_action corresponds to the price after the action was taken.
                # When plotting, you may want to shift(-1) the index actions to better grasp the change in mid price after the action.

                if ep % self.cfg['logging_every'] == 0:
                    print(f"[{self.mode.upper()}][{ep}/{self.episodes}]  Reward={ep_reward:.2f}")

            wandb.finish()

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

            return dic, self.run_id