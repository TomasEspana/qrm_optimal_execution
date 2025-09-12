# import pickle 
# import sys
# import os
# import numpy as np 
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

# from qrm_rl.configs.config import load_config
# from qrm_rl.runner import RLRunner
# from qrm_rl.agents.benchmark_strategies import TWAPAgent, BackLoadAgent, FrontLoadAgent, RandomAgent, ConstantAgent, BimodalAgent, BestVolumeAgent


# if __name__ == "__main__":

#     trader_times = np.array([0., 0., 3., 6., 9., 12., 15., 18., 21.])
#     trader_times = np.array([0., 0., 1., 2., 3., 4., 5., 6., 7.])

#     thetas = np.linspace(0.5, 1.0, 20)
#     theta_reinits = np.linspace(0.5, 1.0, 20)

#     for i, theta in enumerate(thetas):
#         for j, theta_r in enumerate(theta_reinits):

#             config = load_config(theta=theta, theta_r=theta_r, trader_times=trader_times)
#             config['mode'] = 'test' 
#             config['seed'] = 2025
#             config['test_save_memory'] = True # True

#             ### ----------------------###
#             train_run_id = f'heatmap_t_{i}_tr_{j}'
#             config['episodes'] = 20_000 
#             ### ----------------------###

#             runner = RLRunner(config)
#             th = runner.cfg['time_horizon']
#             ii = runner.cfg['initial_inventory']
#             tts = runner.cfg['trader_time_step']
#             actions = runner.cfg['actions']


#             ## === Best Volume - Agent Testing === ###
#             mod = 8
#             runner = RLRunner(config)
#             agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
#             runner.agent = agent
#             dic, run_id = runner.run()
#             with open(f'data_wandb/dictionaries/best_volume_{mod}_{train_run_id}.pkl', 'wb') as f:
#                 pickle.dump(dic, f)






import pickle 
import sys
import os
import numpy as np 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 


from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product


# Optional: avoid WandB collisions if RLRunner might touch it
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import BestVolumeAgent

def run_one(i, j, theta, theta_r, trader_times, episodes=20_000, mod=8, seed=2025):
    """
    Single job: build config, run RL, and dump results.
    Kept self-contained so it works in a separate process.
    """
    # --- Build config ---
    config = load_config(theta=theta, theta_r=theta_r, trader_times=trader_times)
    config["mode"] = "test"
    config["seed"] = seed
    config["test_save_memory"] = True
    config["episodes"] = episodes

    # --- Runner & agent ---
    runner = RLRunner(config)
    agent  = BestVolumeAgent(fixed_action=-1, modulo=mod)
    runner.agent = agent

    # --- Run ---
    dic, run_id = runner.run()   # assumes (results_dict, run_id)

    # --- Persist ---
    train_run_id = f"heatmap_t_{i}_tr_{j}"
    out_dir = "data_wandb/dictionaries"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"best_volume_{mod}_{train_run_id}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(dic, f)

    return (i, j, out_path)




if __name__ == "__main__":


    trader_times = np.array([0., 0., 3., 6., 9., 12., 15., 18., 21.]) # mod = 8
    trader_times = np.array([0., 0., 3.]) # mod = 8
    # trader_times = np.array([0., 0., 1., 2., 3., 4., 5., 6., 7.]) # mod = 8
    # trader_times = np.array([0., 0., 1.0, 30.0, 100.0])

    thetas = np.linspace(0.5, 1.0, 20)
    theta_reinits = np.linspace(0.5, 1.0, 20)

    # Build all jobs
    jobs = [(i, j, theta, theta_r, trader_times)
            for i, theta in enumerate(thetas)
            for j, theta_r in enumerate(theta_reinits)]

    # Sensible default: use (num_cores - 1) workers, at least 1
    max_workers = 4

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, *args) for args in jobs]
        for fut in as_completed(futures):
            try:
                i, j, path = fut.result()
                print(f"✓ Finished (i={i}, j={j}) → {path}")
                results.append((i, j, path))
            except Exception as e:
                # Don't crash the whole pool: report and continue
                print(f"✗ Job failed: {e}")

    # 'results' contains all (i, j, output_path)
    print(f"Done. {len(results)}/{len(jobs)} jobs completed.")
