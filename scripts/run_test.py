import pickle 
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import TWAPAgent, BackLoadAgent, FrontLoadAgent, RandomAgent, ConstantAgent, BimodalAgent, BestVolumeAgent

if __name__ == "__main__":

    config = load_config()
    config['mode'] = 'test' 
    config['seed'] = 2025
    config['test_save_memory'] = True

    ### ----------------------###
    train_run_id = 'k3631p6d'
    config['episodes'] = 20_000
    ### ----------------------###

    runner = RLRunner(config)
    th = runner.cfg['time_horizon']
    ii = runner.cfg['initial_inventory']
    tts = runner.cfg['trader_time_step']
    actions = runner.cfg['actions']


    ### === DDQN Agent Testing === ###
    runner = RLRunner(config, load_model_path=f'/scratch/network/te6653/qrm_optimal_execution/save_model/ddqn_{train_run_id}.zip')
    dic, run_id = runner.run(agent_info='DQN')
    with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/ddqn_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    # # ### === Front Load Agent Testing === ###
    # runner = RLRunner(config)
    # agent = FrontLoadAgent(fixed_action=-1)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/front_load_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # # ### === Front Load Agent Testing === ###
    # runner = RLRunner(config)
    # agent = FrontLoadAgent(fixed_action=1)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/front_load_half_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ## === Best Volume - Agent Testing === ###
    # mod = 2
    # runner = RLRunner(config)
    # agent = BestVolumeAgent(fixed_action=1, modulo=mod)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/best_volume_half_{mod}_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ## === Best Volume - Agent Testing === ###
    # mod = 3
    # runner = RLRunner(config)
    # agent = BestVolumeAgent(fixed_action=1, modulo=mod)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/best_volume_half_{mod}_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)


    # ## === Best Volume - Agent Testing === ###
    # mod = 2
    # runner = RLRunner(config)
    # agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/best_volume_{mod}_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)


    # ## === Best Volume - Agent Testing === ###
    # mod = 3
    # runner = RLRunner(config)
    # agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/best_volume_{mod}_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)


    # ## === Best Volume - Agent Testing === ###
    # mod = 4
    # runner = RLRunner(config)
    # agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/best_volume_{mod}_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)


    # ### === TWAP Agent Testing === ###
    # runner = RLRunner(config)
    # agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/twap_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

