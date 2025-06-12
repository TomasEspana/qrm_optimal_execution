import numpy as np
import torch 
import torch.nn as nn
from numba import njit
import torch.optim as optim
from collections import deque, namedtuple
import wandb
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from contextlib import nullcontext
from collections import defaultdict

from simulate_qrm_numba import IntensityTable, sample_stationary_lob, simulate_QRM_jit, update_LOB



# === Example usage ===



if __name__ == "__main__":

    ## THREE THINGS TO CHECK
    # 1. Neural Network Achitecture
    # 1'. Target update frequency: episodes or steps ? 
    # 2. Pseudo epsilon-greedy policy
    # 3. Config parameters


    ### ==== TEST ===== ###
    train_run_id = 't40nwadb'
    config['episodes'] = 10_000
    config['mode'] = 'test'
    config['seed'] = 2025
    config['exec_security_margin'] = 1

    config['memory_capacity'] = 2_000
    config['proba_0'] = 0.2
    config['price_offset'] = 0.0
    config['price_std'] = 0.3
    config['vol_offset'] = 5 #4
    config['vol_std'] = 4 #3.5
    config['target_update_freq'] = 1

    config['history_size'] = 5
    config['state_dim'] = 4 * config['history_size'] + 2
    config['prop_greedy_eps'] = 0.5
    config['alpha_ramp'] = 15
    config['risk_aversion'] = 0.5 # 0.1
    # config['final_penalty'] = 0.1
    # config['memory_capacity'] = 10_000
    # config['batch_size'] = 128
    # config['unif_deter_strats'] = True
    # config['prop_deter_strats'] = config['prop_greedy_eps'] / 5

    runner = RLRunner(config, load_model_path=f'save_model/ddqn_{train_run_id}.pth')

    th = runner.cfg['time_horizon']
    ii = runner.cfg['initial_inventory']
    tts = runner.cfg['trader_time_step']
    max_action = max(runner.cfg['actions'])
    final_is = {}
    

    ### === DDQN Agent Testing === ###
    dic, run_id = runner.run()
    final_is['DDQN'] = dic['final_is']
    with open(f'data_wandb/dictionaries/ddqn_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === TWAP Agent Testing === ###
    runner = RLRunner(config)
    agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
    actions = agent.actions
    actions[-1] += 1 # avoid remaining inventory because of the QRM liquidity constraints
    agent.actions = actions

    runner.agent = agent
    dic, run_id = runner.run()
    final_is['TWAP'] = dic['final_is']
    with open(f'data_wandb/dictionaries/twap_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === Back Load Agent Testing === ###
    runner = RLRunner(config)
    agent = BackLoadAgent(time_horizon=th, initial_inventory=ii, 
                          trader_time_step=tts, fixed_action=max_action, security_margin=2)
    
    runner.agent = agent
    dic, run_id = runner.run()
    final_is['Back Load'] = dic['final_is']
    with open(f'data_wandb/dictionaries/back_load_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === Front Load Agent Testing === ###
    runner = RLRunner(config)
    agent = FrontLoadAgent(fixed_action=max_action)
    runner.agent = agent
    dic, run_id = runner.run()
    final_is['Front Load'] = dic['final_is']
    with open(f'data_wandb/dictionaries/front_load_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === Front Load Agent Testing === ###
    runner = RLRunner(config)
    agent = FrontLoadAgent(fixed_action=1)
    runner.agent = agent
    dic, run_id = runner.run()
    final_is['Front Load - 1'] = dic['final_is']
    with open(f'data_wandb/dictionaries/front_load_1_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)


    # === Plot Implementation Shortfall ===
    plt.figure(figsize=(7, 5))
    maxi = max(
        max(np.abs(np.max(values)), np.abs(np.min(values)))
        for values in final_is.values()
    )
    x = np.linspace(-maxi, maxi, 1000)
    for key, values in final_is.items():
        kde = gaussian_kde(values)
        y_kde = kde(x)
        plt.plot(x, y_kde, label=key)

    plt.xlabel('Implementation Shortfall')
    plt.ylabel('Density')
    plt.title('')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/implementation_shortfall/{train_run_id}.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
