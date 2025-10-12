import pickle 
import sys
import os
import numpy as np
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import TWAPAgent, BackLoadAgent, FrontLoadAgent, RandomAgent, ConstantAgent, BimodalAgent, BestVolumeAgent


if __name__ == "__main__":

    start = time()

    thetas = np.linspace(0.5, 1.0, 10)
    rel_mean_is_arr = np.zeros((len(thetas), len(thetas))) 
    ranking_arr = np.zeros((len(thetas), len(thetas), 4)) # rl, fl, bv4, twap

    for i, theta in enumerate(thetas):
        theta = thetas[i]

        for j, theta_r in enumerate(thetas):
            theta_r = thetas[j]

            config = load_config(theta=theta, theta_r=theta_r)
            config['mode'] = 'test' 
            config['seed'] = 2025
            config['test_save_memory'] = True
            config['logging'] = False

            # MAKE SURE ACTIONS ARE 0, 1.0 in default.yaml !!!!!!!!!!!!!!

            ### ----------------------###
            train_run_id = 'k3631p6d'
            config['episodes'] = 10_000
            ### ----------------------###

            runner = RLRunner(config)
            th = runner.cfg['time_horizon']
            ii = runner.cfg['initial_inventory']
            tts = runner.cfg['trader_time_step']
            del runner
            

            ### === DDQN Agent Testing === ###
            runner = RLRunner(config, load_model_path=f'/scratch/network/te6653/qrm_optimal_execution/save_model/ddqn_{train_run_id}.zip')
            dic, run_id = runner.run(agent_info='DQN')
            mi_rl = np.mean(np.array(dic['final_is']))

            # ### === Front Load Agent Testing === ###
            runner = RLRunner(config)
            agent = FrontLoadAgent(fixed_action=-1)
            runner.agent = agent
            dic, run_id = runner.run()
            mi_fl = np.mean(np.array(dic['final_is']))

            ## === Best Volume - Agent Testing === ###
            mod = 4
            runner = RLRunner(config)
            agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
            runner.agent = agent
            dic, run_id = runner.run()
            mi_bv4 = np.mean(np.array(dic['final_is']))

            ### === TWAP Agent Testing === ###
            runner = RLRunner(config)
            agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
            runner.agent = agent
            dic, run_id = runner.run()
            mi_twap = np.mean(np.array(dic['final_is']))


            # Comparing and saving results
            max_benchmark = max(mi_fl, mi_bv4, mi_twap)
            rel_mean_is = (mi_rl - max_benchmark) / abs(max_benchmark)
            rel_mean_is_arr[i, j] = rel_mean_is
            ranking_arr[i, j] = np.argsort(np.argsort([mi_rl, mi_twap, mi_bv4, mi_fl])) # 3 is best, 0 is worst. we are expecting [3,2,1,0].
            print(f"theta: {theta}, theta_r: {theta_r}, rel_mean_is: {rel_mean_is}")

    np.save(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/relative_is_heatmap_{train_run_id}.npy', rel_mean_is_arr)
    np.save(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/ranking_is_heatmap_{train_run_id}.npy', ranking_arr)
    print(f"Total time: {(time() - start)/60} min")


        # ## === Best Volume - Agent Testing === ###
        # mod = 2
        # runner = RLRunner(config)
        # agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
        # runner.agent = agent
        # dic, run_id = runner.run()
        # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/best_volume_{mod}_{train_run_id}.pkl', 'wb') as f:
        #     pickle.dump(dic, f)


        # ## === Best Volume - Agent Testing === ###
        # mod = 3
        # runner = RLRunner(config)
        # agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
        # runner.agent = agent
        # dic, run_id = runner.run()
        # with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/best_volume_{mod}_{train_run_id}.pkl', 'wb') as f:
        #     pickle.dump(dic, f)


        




# # ======== USER-EDITABLE KNOBS ========
# TRAIN_RUN_ID = "k3631p6d"
# THETAS = np.linspace(0.5, 1.0, 2)  # e.g., [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# SEED = 2025
# PAIR_OUT_DIR = "/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/tmp_pairs"
# MERGE_OUT_DIR = "/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries"
# AUTO_CLEAN_PAIR_FILES_AFTER_MERGE = True  # delete per-pair .npz after merge
# # =====================================

# # ---- your project imports (must exist in your env) ----
# # from your_project import RLRunner, FrontLoadAgent, BestVolumeAgent, TWAPAgent, load_config

# # If they are in the same package as your original script, keep these:
# # from your_module import RLRunner, FrontLoadAgent, BestVolumeAgent, TWAPAgent, load_config

# # ----------------- utilities -----------------
# def ensure_dir(p):
#     os.makedirs(p, exist_ok=True)

# def idx_to_pair(idx, n):
#     """Map linear index -> (i, j) for n x n grid, row-major."""
#     i = idx // n
#     j = idx % n
#     return i, j

# def pair_to_idx(i, j, n):
#     return i * n + j

# def _roundf(x, nd=12):
#     return float(np.round(float(x), nd))

# def run_one_pair(theta, theta_r, train_run_id, out_dir, seed=2025):
#     """
#     Runs all agents for a single (theta, theta_r) pair and writes a per-pair .npz file.
#     This function is race-free and safe to run concurrently across many jobs.
#     """
#     # --- config setup (as in your code) ---
#     config = load_config(theta=theta, theta_r=theta_r)
#     config["mode"] = "test"
#     config["seed"] = seed
#     config["test_save_memory"] = True
#     config["episodes"] = 20_000  # your original setting

#     # Ensure actions are {0, 1.0} in default.yaml as you noted.

#     # Extract cfg bits you need later (and release early)
#     tmp_runner = RLRunner(config)
#     th = tmp_runner.cfg["time_horizon"]
#     ii = tmp_runner.cfg["initial_inventory"]
#     tts = tmp_runner.cfg["trader_time_step"]
#     del tmp_runner

#     # === DDQN Agent Testing ===
#     runner = RLRunner(
#         config,
#         load_model_path=f"/scratch/network/te6653/qrm_optimal_execution/save_model/ddqn_{train_run_id}.zip",
#     )
#     dic, run_id = runner.run(agent_info="DQN")
#     mi_rl = float(np.mean(np.array(dic["final_is"])))

#     # === Front Load Agent Testing ===
#     runner = RLRunner(config)
#     agent = FrontLoadAgent(fixed_action=-1)
#     runner.agent = agent
#     dic, run_id = runner.run()
#     mi_fl = float(np.mean(np.array(dic["final_is"])))

#     # === Best Volume Agent Testing ===
#     mod = 4
#     runner = RLRunner(config)
#     agent = BestVolumeAgent(fixed_action=-1, modulo=mod)
#     runner.agent = agent
#     dic, run_id = runner.run()
#     mi_bv4 = float(np.mean(np.array(dic["final_is"])))

#     # === TWAP Agent Testing ===
#     runner = RLRunner(config)
#     agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
#     runner.agent = agent
#     dic, run_id = runner.run()
#     mi_twap = float(np.mean(np.array(dic["final_is"])))

#     # Compare + package results
#     max_benchmark = max(mi_fl, mi_bv4, mi_twap)
#     rel_mean_is = (mi_rl - max_benchmark) / abs(max_benchmark)

#     # order: [mi_rl, mi_twap, mi_bv4, mi_fl]  -> bigger rank is better (0..3)
#     order = np.argsort(np.argsort([mi_rl, mi_twap, mi_bv4, mi_fl]))

#     # Save per-pair (no shared arrays -> no races)
#     ensure_dir(out_dir)
#     # Use rounded values in filename to avoid FP noise
#     fname = f"pair_theta={_roundf(theta):.10g}_thetar={_roundf(theta_r):.10g}.npz"
#     np.savez_compressed(
#         os.path.join(out_dir, fname),
#         theta=float(theta),
#         theta_r=float(theta_r),
#         mi_rl=mi_rl,
#         mi_fl=mi_fl,
#         mi_bv4=mi_bv4,
#         mi_twap=mi_twap,
#         rel_mean_is=float(rel_mean_is),
#         ranking=order.astype(np.int16),
#     )
#     print(f"[OK] theta={theta}, theta_r={theta_r}, rel_mean_is={rel_mean_is:.6g}")

# def merge_results(thetas, pair_dir, out_dir, train_run_id):
#     """
#     Assembles rel_mean_is_arr [n,n] and ranking_arr [n,n,4] from per-pair .npz outputs.
#     """
#     n = len(thetas)
#     rel_mean_is_arr = np.zeros((n, n), dtype=np.float64)
#     ranking_arr = np.zeros((n, n, 4), dtype=np.int16)

#     files = [f for f in os.listdir(pair_dir) if f.endswith(".npz")]
#     if not files:
#         raise RuntimeError(f"No .npz files found in {pair_dir}; did the jobs run?")

#     # Build lookup keyed by rounded (theta, theta_r) to be robust to float repr
#     data_map = {}
#     for f in files:
#         d = np.load(os.path.join(pair_dir, f))
#         t = _roundf(d["theta"])
#         tr = _roundf(d["theta_r"])
#         data_map[(t, tr)] = d

#     for i, theta in enumerate(thetas):
#         for j, theta_r in enumerate(thetas):
#             key = (_roundf(theta), _roundf(theta_r))
#             if key not in data_map:
#                 raise KeyError(f"Missing result for theta={theta}, theta_r={theta_r}")
#             d = data_map[key]
#             rel_mean_is_arr[i, j] = float(d["rel_mean_is"])
#             ranking_arr[i, j] = d["ranking"]

#     ensure_dir(out_dir)
#     np.save(os.path.join(out_dir, f"relative_is_heatmap_{train_run_id}.npy"), rel_mean_is_arr)
#     np.save(os.path.join(out_dir, f"ranking_is_heatmap_{train_run_id}.npy"), ranking_arr)
#     print(f"[MERGED] wrote arrays to {out_dir}")

#     if AUTO_CLEAN_PAIR_FILES_AFTER_MERGE:
#         for f in files:
#             try:
#                 os.remove(os.path.join(pair_dir, f))
#             except OSError:
#                 pass

# def main():
#     start = time()
#     thetas = np.array([float(x) for x in THETAS], dtype=float)
#     n = len(thetas)

#     array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
#     if array_id is not None:
#         # ---- array mode: run one pair only ----
#         idx = int(array_id)
#         if idx < 0 or idx >= n * n:
#             raise RuntimeError(f"SLURM_ARRAY_TASK_ID {idx} out of range for n={n} (need 0..{n*n-1})")
#         i, j = idx_to_pair(idx, n)
#         theta = thetas[i]
#         theta_r = thetas[j]
#         run_one_pair(theta, theta_r, TRAIN_RUN_ID, PAIR_OUT_DIR, seed=SEED)
#         # In array mode we do NOT merge here; run merge once after all jobs finish.
#     else:
#         # ---- local/serial mode: sweep all and merge ----
#         for i, theta in enumerate(thetas):
#             for j, theta_r in enumerate(thetas):
#                 run_one_pair(theta, theta_r, TRAIN_RUN_ID, PAIR_OUT_DIR, seed=SEED)
#         merge_results(thetas, PAIR_OUT_DIR, MERGE_OUT_DIR, TRAIN_RUN_ID)

#     print(f"Total wall time: {(time() - start)/60:.2f} min")

# # ================== ENTRY ==================
# if __name__ == "__main__":
#     # imports are resolved at runtime from your environment
#     # (RLRunner, FrontLoadAgent, BestVolumeAgent, TWAPAgent, load_config must be importable)
#     main()