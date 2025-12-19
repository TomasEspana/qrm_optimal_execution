import pickle 
import sys
import os
import numpy as np
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import TWAPAgent, BackLoadAgent, FrontLoadAgent, RandomAgent, ConstantAgent, BimodalAgent, BestVolumeAgent


if __name__ == "__main__":

    start = time()

    thetas = np.linspace(0.5, 1.0, 10)
    rel_mean_is_arr = np.zeros((len(thetas), len(thetas))) 
    ranking_arr = np.zeros((len(thetas), len(thetas), 4)) # rl, fl, bv4, twap

    for i, theta in enumerate(thetas):
        theta = thetas[i]

        for j, theta_r in enumerate(thetas):
            theta_r = thetas[j]

            config = load_config(theta=theta, theta_r=theta_r)
            config['mode'] = 'test' 
            config['seed'] = 2025
            config['test_save_memory'] = True
            config['logging'] = False

            # MAKE SURE ACTIONS ARE 0, 1.0 in default.yaml !!!!!!!!!!!!!!

            ### ----------------------###
            train_run_id = 'k3631p6d'
            config['episodes'] = 10_000
            ### ----------------------###

            runner = RLRunner(config)
            th = runner.cfg['time_horizon']
            ii = runner.cfg['initial_inventory']
            tts = runner.cfg['trader_time_step']
            del runner
            

            ### === DDQN Agent Testing === ###
            runner = RLRunner(config, load_model_path=f'/scratch/network/te6653/qrm_optimal_execution/save_model/ddqn_{train_run_id}.zip')
            dic, run_id = runner.run(agent_info='DQN')
            mi_rl = np.mean(np.array(dic['final_is']))

            # ### === Front Load Agent Testing === ###
            runner = RLRunner(config)
            agent = FrontLoadAgent(fixed_action=-1)
            runner.agent = agent
            dic, run_id = runner.run()
            mi_fl = np.mean(np.array(dic['final_is']))

            ## === Best Volume - Agent Testing === ###
            mod = 4
            runner = RLRunner(config)
            agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
            runner.agent = agent
            dic, run_id = runner.run()
            mi_bv4 = np.mean(np.array(dic['final_is']))

            ### === TWAP Agent Testing === ###
            runner = RLRunner(config)
            agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
            runner.agent = agent
            dic, run_id = runner.run()
            mi_twap = np.mean(np.array(dic['final_is']))


            # Comparing and saving results
            max_benchmark = max(mi_fl, mi_bv4, mi_twap)
            rel_mean_is = (mi_rl - max_benchmark) / abs(max_benchmark)
            rel_mean_is_arr[i, j] = rel_mean_is
            ranking_arr[i, j] = np.argsort(np.argsort([mi_rl, mi_twap, mi_bv4, mi_fl])) # 3 is best, 0 is worst. we are expecting [3,2,1,0].
            print(f"theta: {theta}, theta_r: {theta_r}, rel_mean_is: {rel_mean_is}")

    np.save(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/relative_is_heatmap_{train_run_id}.npy', rel_mean_is_arr)
    np.save(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/ranking_is_heatmap_{train_run_id}.npy', ranking_arr)
    print(f"Total time: {(time() - start)/60} min")


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
