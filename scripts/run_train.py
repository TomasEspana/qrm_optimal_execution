from src.qrm_rl.configs.config import load_config
from src.qrm_rl.runner import RLRunner

if __name__ == "__main__":

    config = load_config("configs/default.yaml")

    config['episodes'] = 10_000
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
    
    runner = RLRunner(config)
    runner.run()