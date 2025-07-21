import yaml
import os
import numpy as np

""" 
    Upload run configuration as dictionary from YAML file.
"""

def load_config(filename="default.yaml"):
    config_path = os.path.join(os.path.dirname(__file__), filename)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    th = config['time_horizon']
    st = config['trader_time_step']
    config['trader_times'] = np.arange(0, th + st, st)
    config['action_dim'] = len(config['actions'])
    if config['basic_state']:
        config['state_dim'] = 3 # Basic state: [inventory, time, mid_price]
    else:
        config['state_dim'] = config['len_state_lob'] * config['history_size'] + 2
    
    config['proba_0'] = 1 / len(config['actions'])

    if config['warmup_steps'] == 0:
        config['warmup_steps'] = config['batch_size']

    # Size pre initialization for LOB
    config['max_events_intra'] = int(200 * config['trader_time_step'])
    config['max_events'] = int((int(config['time_horizon'] / config['trader_time_step']) + 1) * config['max_events_intra'])

    if config['normal_prices']:
        config['file_name'] = 'qrm_paper.npy'
        config['file_name_bid'] = 'qrm_paper.npy'
        config['file_name_ask'] = 'qrm_paper.npy'
        # config['file_name'] = 'aapl_corrected.npy'
        # config['file_name_bid'] = 'aapl_corrected.npy'
        # config['file_name_ask'] = 'aapl_corrected.npy'
    else:
        config['file_name'] = 'aapl_price_down.npy'
        config['file_name_bid'] = 'aapl_price_down_bid.npy'
        config['file_name_ask'] = 'aapl_price_down_ask.npy'

    return config