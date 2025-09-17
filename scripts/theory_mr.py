import pickle 
import sys
import os
import numpy as np 
from pathlib import Path
from datetime import datetime
import gc
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import BestVolumeAgent



""" 
GENERAL DESCRIPTION:
    File used to generate the heatmap of mean reversion after buying the best ask
    for different values of theta and theta_reinit in PHYSICAL TIME. 
    This is quite slow to run (much slower than in event time).
    The main parameters to set are: 
        - trader_times ([0., 0., 1.0, etc], the two zeros at the beginning to force immediate buy)
        - episodes: number of repetitions for each (theta, theta_reinit) pair
        - nb_grid: number of points in the grid for theta and theta_reinit (total jobs = nb_grid^2)
        - out_path to save the results
    BE CAREFUL: 
        - with the path to save the results (make sure the directory exists and you have write permission)
"""


def build_runner(theta, theta_r, time_horizon, trader_times, longest_step, nb_steps,
            logging=False, episodes=20_000, mod=8, seed=2025):
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

        return runner


def main():
    # ----------------------------
    trader_times = np.array([0., 0., 0.25, 0.5, 0.75, 1.0, 3.0])
    diff = np.diff(trader_times)
    longest_step = np.max(diff) if len(diff) > 0 else trader_times[0]
    time_horizon = np.max(trader_times)
    nb_steps = len(trader_times) - 1
    mod = len(trader_times) + 2
    # ----------------------------
    episodes = 10_000
    nb_grid = 3
    thetas = np.linspace(0.5, 1.0, nb_grid, dtype=float)
    theta_reinits = np.linspace(0.5, 1.0, nb_grid, dtype=float)
    arr_all_runs = np.empty((nb_grid, nb_grid, episodes, len(trader_times)-1), dtype=float)

    jobs = [(i, j, theta, theta_r, time_horizon, trader_times, longest_step, nb_steps)
            for i, theta in enumerate(thetas)
            for j, theta_r in enumerate(theta_reinits)]

    total = len(jobs)
    done = 0
    failures = 0
    start_time = datetime.now()
    print(f"Starting sweep over {total} jobs at {start_time}.\n")


    for k, args in enumerate(jobs, start=1):
        i, j, *rest = args
        theta, theta_r, trader_times = rest[0], rest[1], rest[3]
        runner = None  # Initialize to None
        try:
            runner = build_runner(*rest, logging=False, episodes=episodes, mod=mod, seed=2025)
            dic, _ = runner.run()
            dic = dic['mid_prices_events']
            for l in range(len(dic)):
                arr_all_runs[i,j,l,:] = dic[l]

            done += 1
            print(f"✓ [{k}/{total}] Finished (i={i}, j={j}, θ={theta:.4f}, θ_r={theta_r:.4f})", flush=True)
        except Exception as e:
            failures += 1
            print(f"✗ [{k}/{total}] Job failed (i={i}, j={j}, θ={theta:.4f}, θ_r={theta_r:.4f}): {e}", flush=True)
        finally:
            # More aggressive cleanup
            if runner is not None:
                try:
                    # Clean up runner's internal state
                    if hasattr(runner, 'env'):
                        del runner.env
                    if hasattr(runner, 'agent'):
                        del runner.agent
                    del runner
                except Exception:
                    pass
            
            # Force garbage collection
            gc.collect()

            # Periodic memory monitoring
            if k % 10 == 0:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"Memory usage: {memory_mb:.1f} MB", flush=True)

        if k % 50 == 0 or k == total:
            pct = done / total * 100.0
            print(f"Progress: {done}/{total} ({pct:.2f}%) | Failures: {failures}", flush=True)
    
    # --- Save file ---
    out_dir = Path("/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episodes}_runs.npz"
    np.savez_compressed(out_path, arr=arr_all_runs)

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nDone. {done}/{total} ({done/total*100:.2f}%) completed, {failures} failed.")
    print(f"Started: {start_time} | Ended: {end_time} | Elapsed: {elapsed}")



if __name__ == "__main__":
    main()