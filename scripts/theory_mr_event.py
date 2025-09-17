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
from qrm_core.sampling import sample_stationary_lob, update_LOB
from qrm_core.engine import simulate_QRM_jit
from qrm_core.intensity import IntensityTable


""" 
GENERAL DESCRIPTION:
    File used to generate the heatmap of mean reversion after buying the best ask
    for different values of theta and theta_reinit in EVENT TIME. 
    This is very quick to run (much quicker than in physical time).
    The main parameters to set are: 
        - max_nb_events: number of events to log after buying the best ask
        - episodes: number of repetitions for each (theta, theta_reinit) pair
        - nb_grid: number of points in the grid for theta and theta_reinit (total jobs = nb_grid^2)
        - one_spread: if True, force the initial LOB to have one spread
    BE CAREFUL: 
        - with the path to save the results (make sure the directory exists and you have write permission)
"""



def main():
    # ----------------------------
    max_nb_events = 75
    episodes = 10000
    nb_grid = 60
    one_spread = False
    thetas = np.linspace(0.5, 1.0, nb_grid, dtype=float)
    theta_reinits = np.linspace(0.5, 1.0, nb_grid, dtype=float)
    arr_all_runs = np.empty((nb_grid, nb_grid, episodes, max_nb_events), dtype=float)

    K = 3
    t0 = 0.
    p_ref = 100.005
    tick = 0.01
    aes = [836, 1068, 1069]
    inv_bid = np.load('calibration_data/invariant_distribution/qrm_paper.npy')
    inv_ask = np.load('calibration_data/invariant_distribution/qrm_paper.npy')
    inten_arr = np.load('calibration_data/intensity_table/qrm_paper.npy')
    K, Q1, *_ = inten_arr.shape
    inten_table = IntensityTable(max_depth=K, max_queue=Q1-1)
    inten_table._data = inten_arr
    rate_int_all = np.transpose(inten_table._data, (2,0,1,3)).copy()

    jobs = [(i, j, theta, theta_r)
            for i, theta in enumerate(thetas)
            for j, theta_r in enumerate(theta_reinits)]

    total = len(jobs)
    done = 0
    failures = 0
    start_time = datetime.now()
    print(f"Starting sweep over {total} jobs at {start_time}.\n")


    for k, args in enumerate(jobs, start=1):
        i, j, theta, theta_r = args

        try:

            for l in range(episodes):
                
                # initial lob state
                lob0 = np.empty(2*K, np.int8)
                if one_spread:
                    while True:
                        lob0[:K]   = sample_stationary_lob(inv_bid, np.empty((0,), np.int8))
                        lob0[K:] = sample_stationary_lob(inv_ask, np.empty((0,), np.int8))
                        bid_idx = next((i for i in range(K) if lob0[i]>0), None)
                        ask_idx = next((i for i in range(K, 2*K) if lob0[i]>0), None)
                        if bid_idx is not None and ask_idx is not None and (ask_idx - bid_idx) == K:
                            break
                
                else:
                    lob0[:K]   = sample_stationary_lob(inv_bid, np.empty((0,), np.int8))
                    lob0[K:] = sample_stationary_lob(inv_ask, np.empty((0,), np.int8))
                
                # simulate buying best ask
                ask_idx = next((i for i in range(K, 2*K) if lob0[i]>0), None) # identify best ask
                if ask_idx is None:
                    raise ValueError("Sampled empty LOB")
                lob0[ask_idx] = 0 # buy best ask
                new_pmid, new_pref, state, redrawn = update_LOB( K, p_ref, lob0, 1,
                                                              theta, theta_r, tick,
                                                              inv_bid, inv_ask, aes)
                
                    
                times, p_mids, p_refs, sides, depths, events, redrawns, states = simulate_QRM_jit(t0,
                                                                                            new_pmid,
                                                                                            new_pref,
                                                                                            state,
                                                                                            rate_int_all,
                                                                                            tick,
                                                                                            theta,
                                                                                            theta_r,
                                                                                            1,
                                                                                            inv_bid,
                                                                                            inv_ask, 
                                                                                            max_nb_events+2, 
                                                                                            aes, 
                                                                                            max_nb_events=max_nb_events
                                                                                            )

                arr_all_runs[i,j,l,:] = p_mids

            done += 1
            print(f"✓ [{k}/{total}] Finished (i={i}, j={j}, θ={theta:.4f}, θ_r={theta_r:.4f})", flush=True)

        except Exception as e:
            failures += 1
            print(f"✗ [{k}/{total}] Job failed (i={i}, j={j}, θ={theta:.4f}, θ_r={theta_r:.4f}): {e}", flush=True)

        if k % 50 == 0 or k == total:
            pct = done / total * 100.0
            print(f"Progress: {done}/{total} ({pct:.2f}%) | Failures: {failures}", flush=True)
    
    # --- Save file ---
    out_dir = Path("/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episodes}_runs_event_time_long.npz"
    np.savez_compressed(out_path, arr=arr_all_runs)

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nDone. {done}/{total} ({done/total*100:.2f}%) completed, {failures} failed.")
    print(f"Started: {start_time} | Ended: {end_time} | Elapsed: {elapsed}")



if __name__ == "__main__":
    main()