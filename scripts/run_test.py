import pickle 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import TWAPAgent, POPVAgent

"""
    GENERAL DESCRIPTION:
        Script to run tests for different agents (DDQN, TWAP, POPV) using a pre-trained model.
"""


if __name__ == "__main__":

    config = load_config()
    config['mode'] = 'test' 
    config['seed'] = 2025
    config['test_save_memory'] = True

    ### ---------------------- ###
    train_run_id = '5d_state_3d_action' # see saved_models/ for available run ids
    config['episodes'] = 20_000
    ### ---------------------- ###

    runner = RLRunner(config)
    th = runner.cfg['time_horizon']
    ii = runner.cfg['initial_inventory']
    tts = runner.cfg['trader_time_step']
    actions = runner.cfg['actions']


    ### === DDQN === ###
    runner = RLRunner(config, load_model_path=f'./saved_models/ddqn_{train_run_id}.zip')
    dic, run_id = runner.run(agent_info='DQN')
    with open(f'./results/ddqn_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === TWAP === ###
    runner = RLRunner(config)
    agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
    runner.agent = agent
    dic, run_id = runner.run()
    with open(f'./results/twap_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === POPV === ###
    for k in range(1, 3):
        action_idx = 1 # buy 50% of available volume at best ask at each trader time 
        # Note: make sure action_idx matches config['actions'] (default [0, 0.5, 1.0]) in src/qrm_rl/configs/default.yaml
        runner = RLRunner(config)
        agent = POPVAgent(fixed_action=action_idx, modulo=k)
        runner.agent = agent
        dic, run_id = runner.run()
        if action_idx == 1:
            with open(f'./results/popv{k}_half_{train_run_id}.pkl', 'wb') as f:
                pickle.dump(dic, f)
        elif action_idx == -1:
            with open(f'./results/popv{k}_{train_run_id}.pkl', 'wb') as f:
                pickle.dump(dic, f)