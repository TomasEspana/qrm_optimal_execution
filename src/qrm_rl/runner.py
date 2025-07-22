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
            gamma=config["gamma"],
            target_update_interval=config["target_update_freq"],
            train_freq=1, # update every step
            exploration_initial_eps=config["epsilon_start"],
            exploration_final_eps=config["epsilon_end"],
            exploration_fraction=config["prop_greedy_eps"],
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=self.device,
        )
        self.agent = self.model

    def run(self):
        agent_type = self.agent_name_map.get(type(self.agent), 'Unknown')
        self.run_id = wandb.run.id
        train_mode = (self.mode == 'train')

        if train_mode:
            wandb.run.name = f"{agent_type}_{self.run_id}"
        else:
            wandb.run.name = f"{agent_type}_test_{self.run_id}"

        if not train_mode:
            final_is = []
            lob_dataframe = {}
            times, mid_prices, ref_prices, sides, depths, events, redrawn, states = {}, {}, {}, {}, {}, {}, {}, {}
            actions_taken = {}
            executed_dic = {}
            index_actions = {}

        
        if train_mode:
            total_steps = self.cfg["total_timesteps"]

            callback = CallbackList([
                  WandbCallback(verbose=2),
                InfoLoggerCallback()
            ])

            self.model.learn(total_timesteps=total_steps, callback=callback)
            self.model.save(f"save_model/{agent_type}_{self.run_id}.zip")
            wandb.finish()
            return