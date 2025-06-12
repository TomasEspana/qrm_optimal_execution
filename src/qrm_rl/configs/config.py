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

    normal_prices = True
    decreasing_prices = False
    if normal_prices:
        config['file_name'] = 'aapl_corrected.npy'
        config['file_name_bid'] = 'aapl_corrected.npy'
        config['file_name_ask'] = 'aapl_corrected.npy'
    if decreasing_prices:
        config['file_name'] = 'aapl_price_down.npy'
        config['file_name_bid'] = 'aapl_price_down_bid.npy'
        config['file_name_ask'] = 'aapl_price_down_ask.npy'

    return config