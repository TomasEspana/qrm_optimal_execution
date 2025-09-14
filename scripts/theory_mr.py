import pickle 
import sys
import os
import numpy as np 
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 


from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product


# Optional: avoid WandB collisions if RLRunner might touch it
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import BestVolumeAgent

def run_one(i, j, theta, theta_r, trader_times, logging=False, episodes=10_000, mod=8, seed=2025):
    """
    Single job: build config, run RL, and dump results.
    Kept self-contained so it works in a separate process.
    """
    # --- Build config ---
    config = load_config()
    config["theta"] = theta
    config["theta_reinit"] = theta_r
    config["trader_times"] = trader_times
    config["time_horizon"] = np.max(trader_times)
    config["mode"] = "test"
    config["seed"] = seed
    config["test_save_memory"] = True
    config["episodes"] = episodes
    config['logging'] = logging

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


    trader_times = np.array([0., 0., 0.25, 0.5, 0.75, 1.0, 3.0])  # mod = 8
    nb_grid = 20
    thetas = np.linspace(0.6, 1.0, nb_grid)
    theta_reinits = np.linspace(0.5, 1.0, nb_grid)

    # --- simple double for loop ---
    total = len(thetas) * len(theta_reinits)
    done = 0
    results = []  # (i, j, path)

    t0 = time.time()
    for i, theta in enumerate(thetas):
        for j, theta_r in enumerate(theta_reinits):
            try:
                # run_one must return (i, j, path) or similar
                i_out, j_out, path = run_one(i, j, theta, theta_r, trader_times)
                print(f"✓ Finished (i={i_out}, j={j_out}) → {path}\n", flush=True)
                results.append((i_out, j_out, path))
            except Exception as e:
                print(f"✗ Job failed at (i={i}, j={j}): {e}", flush=True)
            finally:
                done += 1
                print(f"Progress: {done/total*100:.2f}% jobs completed.", flush=True)

    dt = time.time() - t0
    print(f"Done. {done/total*100:.2f}% jobs completed in {dt:.1f}s.")


    # # MULTIPROCESSING VERSION
    # trader_times = np.array([0., 0., 0.25, 0.5, 0.75, 1.0, 3.0]) # mod = 8
    # max_workers = 4

    # nb_grid = 40
    # thetas = np.linspace(0.6, 1.0, nb_grid)
    # theta_reinits = np.linspace(0.6, 1.0, nb_grid)

    # # Build all jobs
    # jobs = [(i, j, theta, theta_r, trader_times)
    #         for i, theta in enumerate(thetas)
    #         for j, theta_r in enumerate(theta_reinits)]

    # done = 0
    # with ProcessPoolExecutor(max_workers=max_workers) as ex:
    #     futures = [ex.submit(run_one, *args) for args in jobs]
    #     for fut in as_completed(futures):
    #         try:
    #             i, j, path = fut.result()
    #             print(f"✓ Finished (i={i}, j={j}) → {path} \n", flush=True)
    #             done += 1
    #             print(f"Progress: {done/len(jobs)*100:.2f}% jobs completed.", flush=True)
    #         except Exception as e:
    #             # Don't crash the whole pool: report and continue
    #             print(f"✗ Job failed: {e}")

    # print(f"Done. {done/len(jobs)*100:.2f}% jobs completed.")

