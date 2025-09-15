import pickle 
import sys
import os
import numpy as np 
from pathlib import Path
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import BestVolumeAgent


try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False



def run_one(i, j, theta, theta_r, time_horizon, trader_times, longest_step, nb_steps,
            logging=False, episodes=20_000, mod=8, seed=2025):
    """
    Single job: build config, run RL, and dump results.
    Returns (i, j, out_path) on success.
    """
    # --- Build config ---
    # be careful: parameters may need to be overwritten before load_config if they affect derived params
    config = load_config(time_horizon=time_horizon, longest_step=longest_step, nb_steps=nb_steps)
    config["theta"] = float(theta)
    config["theta_reinit"] = float(theta_r)
    config["trader_times"] = trader_times
    config["mode"] = "test"
    config["seed"] = int(seed)
    config["test_save_memory"] = True
    config["episodes"] = int(episodes)
    config["logging"] = bool(logging)

    # --- Runner & agent ---
    runner = RLRunner(config)
    agent  = BestVolumeAgent(fixed_action=-1, modulo=mod)
    runner.agent = agent

    # --- Run ---
    dic, run_id = runner.run()
    dic = dic['mid_prices_events']
    arr_prices = np.empty((len(dic), len(trader_times)-1))
    for k in range(len(dic)):
        arr_prices[k,:] = dic[k]

    # --- Folder Path ---
    train_run_id = f"heatmap_t_{i}_tr_{j}"
    out_dir = Path("/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_volume_{mod}_{train_run_id}.npy"

    np.save(out_path, arr_prices)

    return (i, j, str(out_path))


def main():
    # ----------------------------
    # GPU / runtime housekeeping
    # ----------------------------
    if TORCH_AVAILABLE:
        # Pin to GPU:0 by default (change if needed)
        if torch.cuda.is_available():
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            torch.cuda.set_device(0)
            # Keep CPU libs from oversubscribing
            torch.set_num_threads(1)

    # ----------------------------
    trader_times = np.array([0., 0., 0.25, 0.5, 0.75, 1.0, 3.0, 10., 30.])
    diff = np.diff(trader_times)
    longest_step = np.max(diff) if len(diff) > 0 else trader_times[0]
    time_horizon = np.max(trader_times)
    nb_steps = len(trader_times) - 1
    mod = len(trader_times) + 2
    # ----------------------------
    nb_grid = 2
    thetas = np.linspace(0.5, 1.0, nb_grid, dtype=float)
    theta_reinits = np.linspace(0.5, 1.0, nb_grid, dtype=float)

    jobs = [(i, j, theta, theta_r, time_horizon, trader_times, longest_step, nb_steps)
            for i, theta in enumerate(thetas)
            for j, theta_r in enumerate(theta_reinits)]

    total = len(jobs)
    done = 0
    failures = 0

    start_time = datetime.now()
    print(f"Starting single-GPU sweep over {total} jobs at {start_time}.\n")

    # ----------------------------
    # Run jobs sequentially on one GPU
    # ----------------------------
    for k, args in enumerate(jobs, start=1):
        i, j, theta, theta_r, *_ = args
        try:
            ii, jj, path = run_one(*args, logging=False, episodes=20_000, mod=mod, seed=2025)
            done += 1
            print(f"✓ [{k}/{total}] Finished (i={ii}, j={jj}, θ={theta:.4f}, θ_r={theta_r:.4f}) → {path}")
        except Exception as e:
            failures += 1
            print(f"✗ [{k}/{total}] Job failed (i={i}, j={j}, θ={theta:.4f}, θ_r={theta_r:.4f}): {e}")
        finally:
            # Free GPU memory between runs
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if k % 10 == 0 or k == total:
            pct = done / total * 100.0
            print(f"Progress: {done}/{total} ({pct:.2f}%) | Failures: {failures}")

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nDone. {done}/{total} ({done/total*100:.2f}%) completed, {failures} failed.")
    print(f"Started: {start_time} | Ended: {end_time} | Elapsed: {elapsed}")



if __name__ == "__main__":
    main()