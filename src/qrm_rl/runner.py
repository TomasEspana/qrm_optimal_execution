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
from qrm_rl.callbacks import InfoLoggerCallback

# from contextlib import nullcontext
# from qrm_rl.agents.ddqn import DDQNAgent
# from .market_environment import MarketEnvironment
# from .utils import load_model, save_model


# exploration_mode_dic = {'rl': int(0), 'front_load': int(1), 'back_load': int(2), 'twap': int(3)}

class RLRunner:
    def __init__(self, config, load_model_path=None):
        # Unpack config
        self.cfg = config
        self.mode = config['mode']
        self.episodes = config['episodes']
        self.exec_security_margin = config['exec_security_margin']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.unif_deter_strats = config['unif_deter_strats']
        self.prop_greedy_eps = config['prop_greedy_eps']
        self.prop_deter_strats = config['prop_deter_strats'] 
        self.agent = None
        self.load_model_path = load_model_path
        self.agent_name_map = {
            DQN: 'ddqn',
            PPO: 'ppo',
            TWAPAgent: 'twap', 
            BackLoadAgent: 'back_load',
            FrontLoadAgent: 'front_load', 
            RandomAgent: 'random'
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
            config=config
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
            action_dim=config['action_dim']
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
            policy="MlpPolicy",
            env=self.env,
            learning_rate=config["learning_rate"],
            buffer_size=config["memory_capacity"],
            batch_size=config["batch_size"],
            learning_starts=config["batch_size"],
            gamma=config["gamma"],
            target_update_interval=config["target_update_freq"],
            train_freq=1, # update every step
            gradient_steps=config["gradient_steps"],
            exploration_initial_eps=config["epsilon_start"],
            exploration_final_eps=config["epsilon_end"],
            exploration_fraction=config["prop_greedy_eps"],
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
                  WandbCallback(verbose=2),
                InfoLoggerCallback(self.cfg["action_dim"])
            ])

            self.model.learn(total_timesteps=total_steps, callback=callback)
            self.model.save(f"save_model/{agent_type}_{self.run_id}.zip")
            wandb.finish()
            return
        
        else:
            # ===== TEST MODE =====
            wandb.run.name = f"{agent_type}_test_{self.run_id}"

            # Load SB3 model
            if self.load_model_path is not None:
                self.model = DQN.load(self.load_model_path, env=self.env, device=self.device)
            # Logging
            mid_prices, mid_prices_events, lob_dataframe, actions_taken, executed_dic, index_actions = {}, {}, {}, {}, {}, {}
            final_is = []

            for ep in range(self.episodes):

                actions, executed = [], []
                done, ep_reward = False, 0. 

                obs, _ = self.env.reset()
                idx_actions = [self.unwrap_env(self.env)._env.simulator.step - 1]  # first index
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
                    idx_actions.append(self.unwrap_env(self.env)._env.simulator.step - 1)
                    ep_reward += reward
                    k += 1

                # end-of-episode book-keeping (single ep test; extend if multiple desired)
                unwrapped_env = self.unwrap_env(self.env)
                final_is.append(unwrapped_env._env.final_is)
                actions_taken[ep] = actions
                executed_dic[ep] = executed
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
                "index_actions": index_actions
            }

            return dic, self.run_id