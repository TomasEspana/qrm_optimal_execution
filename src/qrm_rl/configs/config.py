import yaml
import os
import numpy as np

""" 
GENERAL DESCRIPTION:

    Set the parameters of a run.
"""

def load_config(time_horizon=None, longest_step=None, nb_steps=None, filename="default.yaml"):

    # Load base configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), filename)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override time_horizon if provided
    if time_horizon is not None:
        config['time_horizon'] = time_horizon

    # Derived parameters
    config['action_dim'] = len(config['actions'])
    config['total_timesteps'] = config['episodes'] * (config['time_horizon'] // config['trader_time_step'])
    
    # Trader times
    th = config['time_horizon']
    st = config['trader_time_step']
    config['trader_times'] = np.arange(0, th + st, st)

    # State dimension
    if config['basic_state']:
        config['state_dim'] = config['len_basic_state']
    else:
        config['state_dim'] = config['len_state_lob'] * config['history_size'] + 2
    
    # Pre-allocation for LOB events
    if longest_step is not None:
        config['max_events_intra'] = int(config['max_events_per_second'] * longest_step) + 1
    else:
        config['max_events_intra'] = int(config['max_events_per_second'] * config['trader_time_step']) + 1 
    if nb_steps is not None:
        config['max_events'] = int(nb_steps * config['max_events_intra']) + 1
    else:
        config['max_events'] = int((int(config['time_horizon'] / config['trader_time_step']) + 1) * config['max_events_intra'])
    
    # Data files for intensities and invariant distributions
    config['file_name'] = 'qrm_paper.npy'     # 'aapl_corrected.npy'
    config['file_name_bid'] = 'qrm_paper.npy' # 'aapl_corrected.npy'
    config['file_name_ask'] = 'qrm_paper.npy' # 'aapl_corrected.npy'

    # Deprecated parameters
    config['proba_0'] = 1 / len(config['actions'])
    config['alpha_ramp'] = 15

    return config