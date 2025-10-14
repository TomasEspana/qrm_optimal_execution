import sys
import os
import numpy as np 
from pathlib import Path
from datetime import datetime
from numba import njit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import BestVolumeAgent
from qrm_core.sampling import sample_stationary_lob, update_LOB
from qrm_core.engine import simulate_QRM_jit
from qrm_core.intensity import IntensityTable


""" 
GENERAL DESCRIPTION:

    HERE WE TRY TO UNDERSTAND WHY PRICES ALWAYS MEAN REVERT ON THE LONG TERM 
    We anayze how mid price evolves in the middle leaf of Fig. 1

    File used to generate mid price evolution for ONE tuple after buying the best ask
    for different values of theta and theta_reinit in EVENT TIME. 
    Similar to theory_mr_event_heatmap.py but for ONE SINGLE (theta, theta_reinit).
    This is very quick to run (much quicker than in physical time).

    The main parameters to set are: 
        - max_nb_events: number of events to log AFTER buying the best ask
        - episodes: number of repetitions for each (theta, theta_reinit) pair
        - nb_grid: number of points in the grid for theta and theta_reinit (total jobs = nb_grid^2)
        - one_spread: if True, force the initial LOB to have one spread
    BE CAREFUL: 
        - with the path to save the results (make sure the directory exists and you have write permission)
"""



def main():
    # ----------------------------

    leaf = 'middle_leaf' # 'middle_leaf' or 'right_leaf' or 'left_leaf'
    qrm_type = 'ergodic'  # 'qrm_paper' or 'permanent' or 'test' or 'ergodic'

    if qrm_type == 'qrm_paper':
        theta = 0.7
        theta_r = 0.85
    elif qrm_type == 'permanent':
        theta = 0.9
        theta_r = 0.6
    elif qrm_type == 'ergodic':
        theta = 0.
        theta_r = 0.5
    elif qrm_type == 'test':
        theta = 0.5
        theta_r = 0.5
    else:
        raise ValueError("qrm_type must be 'qrm_paper' or 'permanent' or 'test' or 'ergodic'")
    
    file_prefix = f"{leaf}_{qrm_type}"

    if leaf == 'middle_leaf':
        theta_first = 1
        theta_r_first = 0
    elif leaf == 'right_leaf':
        theta_first = 0.
        theta_r_first = 0.5 # anything works
    elif leaf == 'left_leaf':
        theta_first = 1.
        theta_r_first = 1.
    else:
        raise ValueError("leaf must be 'middle_leaf' or 'right_leaf' or 'left_leaf'")

    max_nb_events = 1000
    episodes = 1000000
    one_spread = True
    arr_all_runs = np.empty((episodes, max_nb_events), dtype=float)

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

    jobs = [(0, 0, theta, theta_r)]

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
                        nz = np.flatnonzero(lob0[K:])
                        ask_idx = K + nz[0] if len(nz) > 0 else None
                        ask_idx_second = K + nz[1] if len(nz) > 1 else None
                        if bid_idx == 0 and ask_idx == K and ask_idx_second == (K+1):
                            break
                
                else:
                    lob0[:K]   = sample_stationary_lob(inv_bid, np.empty((0,), np.int8))
                    lob0[K:] = sample_stationary_lob(inv_ask, np.empty((0,), np.int8))
                
                # simulate buying best ask
                ask_idx = next((i for i in range(K, 2*K) if lob0[i]>0), None) # identify best ask
                if ask_idx is None:
                    raise ValueError("Sampled empty LOB")
                lob0[ask_idx] = 0 # buy best ask
                # middle leaf corresponds to theta=1 and theta_r=0
                new_pmid, new_pref, state, redrawn = update_LOB( K, p_ref, lob0, 1,
                                                              theta_first, theta_r_first, tick,
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

                arr_all_runs[l,:] = p_mids

            done += 1
            print(f"✓ [{k}/{total}] Finished (i={i}, j={j}, θ={theta:.4f}, θ_r={theta_r:.4f})", flush=True)

        except Exception as e:
            failures += 1
            print(f"✗ [{k}/{total}] Job failed (i={i}, j={j}, θ={theta:.4f}, θ_r={theta_r:.4f}): {e}", flush=True)

        if k % 50 == 0 or k == total:
            pct = done / total * 100.0
            print(f"Progress: {done}/{total} ({pct:.2f}%) | Failures: {failures}", flush=True)
    
    # --- Save file ---
    # Average across episodes
    arr_all_runs = np.mean(arr_all_runs, axis=0)
    out_dir = Path("/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_prefix}_market_impact.npz"
    np.savez_compressed(out_path, arr=arr_all_runs)

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nDone. {done}/{total} ({done/total*100:.2f}%) completed, {failures} failed.")
    print(f"Started: {start_time} | Ended: {end_time} | Elapsed: {elapsed}")



if __name__ == "__main__":
    np.random.seed(2025)
    @njit
    def _init_numba(seed): np.random.seed(seed)
    _init_numba(2025)

    main()