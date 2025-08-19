import pickle 
import sys
import os
import numpy as np 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import TWAPAgent, BackLoadAgent, FrontLoadAgent, RandomAgent, ConstantAgent, BimodalAgent, BestVolumeAgent


if __name__ == "__main__":

    trader_times = np.array([0., 0., 3., 6., 9., 12., 15., 18., 21.])
    trader_times = np.array([0., 0., 1., 2., 3., 4., 5., 6., 7.])

    thetas = np.linspace(0.5, 1.0, 20)
    theta_reinits = np.linspace(0.5, 1.0, 20)

    for i, theta in enumerate(thetas):
        for j, theta_r in enumerate(theta_reinits):

            config = load_config(theta=theta, theta_r=theta_r, trader_times=trader_times)
            config['mode'] = 'test' 
            config['seed'] = 2025
            config['test_save_memory'] = True # True

            ### ----------------------###
            train_run_id = f'heatmap_t_{i}_tr_{j}'
            config['episodes'] = 20_000 
            ### ----------------------###

            runner = RLRunner(config)
            th = runner.cfg['time_horizon']
            ii = runner.cfg['initial_inventory']
            tts = runner.cfg['trader_time_step']
            actions = runner.cfg['actions']


            ## === Best Volume - Agent Testing === ###
            mod = 8
            runner = RLRunner(config)
            agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
            runner.agent = agent
            dic, run_id = runner.run()
            with open(f'data_wandb/dictionaries/best_volume_{mod}_{train_run_id}.pkl', 'wb') as f:
                pickle.dump(dic, f)