import numpy as np
import pandas as pd

def simulate_QRM(current_LOB: pd.DataFrame,
                     intensity_table: IntensityTable,
                     tick: float,
                     theta: float,
                     theta_reinit: float,
                     time_end: float,
                     inv_dist_file_bid: str,
                     inv_dist_file_ask: str,
                     seed: int
                     ):
    
    # unpack history
    time  = current_LOB['time'].to_numpy()[-1]
    p_mid = current_LOB['p_mid'].to_numpy()[-1]
    p_ref = current_LOB['p_ref'].to_numpy()[-1]

    K, Q = intensity_table.max_depth, intensity_table.max_queue
    cols = [f"q_bid{i+1}" for i in range(K)] + [f"q_ask{i+1}" for i in range(K)]
    state = current_LOB[cols].to_numpy()[-1].astype(int)

    # precompute intensities
    rate_int_all = np.zeros((2, K, Q+1, 3), np.float64)
    for s_i, side in enumerate(['bid', 'ask']):
        for d in range(1, K+1):
            for qsz in range(Q+1):
                ints = intensity_table.get_intensities(d, qsz, side)
                rate_int_all[s_i, d-1, qsz, 0] = ints['limit']
                rate_int_all[s_i, d-1, qsz, 1] = ints['cancel']
                rate_int_all[s_i, d-1, qsz, 2] = ints['market']

    inv_bid = np.load(inv_dist_file_bid)
    inv_ask = np.load(inv_dist_file_ask)

    # all this part up to here can be removed and done only once for after in the RL environment

    (times2, mids2, prefs2,
     sides2, depths2, evs2,
     red2, states2) = simulate_QRM_jit(
        time, p_mid, p_ref, state,
        rate_int_all, tick, theta, theta_reinit,
        time_end, inv_bid, inv_ask, seed
    )

    df = pd.DataFrame({
        'time':    np.array(times2),
        'p_mid':   np.array(mids2),
        'p_ref':   np.array(prefs2),
        'side':    np.array(sides2),
        'depth':   np.array(depths2),
        'event':   np.array(evs2),
        'redrawn': np.array(red2),
    })
    state_arr = np.vstack(states2)
    for j, col in enumerate(cols):
        df[col] = state_arr[:, j]

    # map numeric → strings
    df['side']  = df['side'].map({1:'bid', 2:'ask'})
    df['event'] = df['event'].map({1:'limit', 2:'cancel', 3:'order'})

    # reorder queue columns: q_bidK…q_bid1, then q_ask1…q_askK
    bid_cols = [f"q_bid{i}" for i in range(K, 0, -1)]
    ask_cols = [f"q_ask{i}" for i in range(1, K+1)]
    final_cols = ['time','p_mid','p_ref','side','depth','event','redrawn'] + bid_cols + ask_cols

    # concat with history and reindex
    full = pd.concat([current_LOB, df], ignore_index=True)
    return full[final_cols]
