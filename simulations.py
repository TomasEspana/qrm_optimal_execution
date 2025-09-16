import numpy as np 
import time
from numba import njit
import pandas as pd

from src.qrm_core.engine import simulate_QRM_jit
from src.qrm_core.sampling import sample_stationary_lob
from src.qrm_core.intensity import IntensityTable


def to_dataframe(times, p_mids, p_refs, sides, depths, events, redrawn, states, K):
    # Labels → categoricals (cheap)
    side_labels  = np.array(['', 'bid', 'ask'], dtype=object)
    event_labels = np.array(['', 'limit', 'cancel', 'market', 'trader'], dtype=object)
    side_cat  = pd.Categorical(side_labels[sides],  categories=['bid','ask'])
    event_cat = pd.Categorical(event_labels[events], categories=['limit','cancel','market','trader'])

    # Base df
    df = pd.DataFrame({
        'time':    times,
        'p_mid':   p_mids,
        'p_ref':   p_refs,
        'side':    side_cat,
        'depth':   depths,
        'event':   event_cat,
        'redrawn': redrawn.astype(bool, copy=False),
    }, copy=False)

    # --- Build q_* without hstack/copies ---
    # Make a DataFrame that *views* `states` (avoid copies if it's a 2D contiguous array)
    q_df = pd.DataFrame(states, copy=False)

    # Column order: bids reversed, asks as-is (logical reindex; inexpensive)
    order = list(range(K-1, -1, -1)) + list(range(K, 2*K))
    q_df = q_df.iloc[:, order]  # reorders column labels; usually no data copy

    # Name columns
    bid_names = [f"q_bid{i+1}" for i in range(K)][::-1]  # q_bidK ... q_bid1
    ask_names = [f"q_ask{i+1}" for i in range(K)]        # q_ask1 ... q_askK
    q_df.columns = bid_names + ask_names

    # Concatenate once (still a copy, but only once and without building q_block first)
    out = pd.concat([df, q_df], axis=1, copy=False)
    return out


def to_dataframe_short(times, p_mids, sides, events, states, K):
    # Labels → categoricals (cheap)
    side_labels  = np.array(['', 'bid', 'ask'], dtype=object)
    event_labels = np.array(['', 'limit', 'cancel', 'market', 'trader'], dtype=object)
    side_cat  = pd.Categorical(side_labels[sides],  categories=['bid','ask'])
    event_cat = pd.Categorical(event_labels[events], categories=['limit','cancel','market','trader'])

    # Base df
    df = pd.DataFrame({
        'time':    pd.Series(times, dtype=np.float64, copy=False),
        'p_mid':   p_mids,
        'side':    side_cat,
        'event':   event_cat
        }, copy=False)

    q_df = pd.DataFrame(states, copy=False)

    order = list(range(K-1, -1, -1)) + list(range(K, 2*K))
    q_df = q_df.iloc[:, order] 

    # Name columns
    bid_names = [f"q_bid{i+1}" for i in range(K)][::-1]  # q_bidK ... q_bid1
    ask_names = [f"q_ask{i+1}" for i in range(K)]        # q_ask1 ... q_askK
    q_df.columns = bid_names + ask_names

    out = pd.concat([df, q_df], axis=1, copy=False)
    return out


if __name__ == "__main__":

    np.random.seed(42)
    @njit
    def _init_numba(seed): np.random.seed(seed)
    _init_numba(42)

    aes = [836, 1068, 1069]
    t0 = 0.
    p_mid = 100.005
    p_ref = 100.005

    # Build intensity table
    inten_arr = np.load('calibration_data/intensity_table/qrm_paper.npy')
    K, Q1, *_ = inten_arr.shape
    inten_table = IntensityTable(max_depth=K, max_queue=Q1-1)
    inten_table._data = inten_arr
    rate_int_all = np.transpose(inten_table._data, (2,0,1,3)).copy()

    # initial lob state (force one spread)
    inv_bid = np.load('calibration_data/invariant_distribution/qrm_paper.npy')
    inv_ask = np.load('calibration_data/invariant_distribution/qrm_paper.npy')
    state = np.empty(2*K, np.int8)
    one_spread = False
    if one_spread:
        while True:
            state[:K]   = sample_stationary_lob(inv_bid, np.empty((0,), np.int8))
            state[K:] = sample_stationary_lob(inv_ask, np.empty((0,), np.int8))
            bid_idx = next((i for i in range(K) if state[i]>0), None)
            ask_idx = next((i for i in range(K, 2*K) if state[i]>0), None)
            if bid_idx is not None and ask_idx is not None and (ask_idx - bid_idx) == K:
                break

    else:
        state[:K]   = sample_stationary_lob(inv_bid, np.empty((0,), np.int8))
        state[K:] = sample_stationary_lob(inv_ask, np.empty((0,), np.int8))

    tick = 0.01
    theta = 0.9
    theta_reinit = 0.6
    time_end = 100
    max_events_intra = 12 * int(time_end)

    # warm-up: (compiles the JIT) simulate only 0.5 seconds
    simulate_QRM_jit(t0,p_mid, p_ref,state,rate_int_all, tick, theta, theta_reinit, 1., inv_bid, inv_ask,  30,  aes)
    
    ## real run
    start = time.time()
    times, p_mids, p_refs, sides, depths, events, redrawns, states = simulate_QRM_jit(t0,
                                                                                        p_mid,
                                                                                        p_ref,
                                                                                        state,
                                                                                        rate_int_all,
                                                                                        tick,
                                                                                        theta,
                                                                                        theta_reinit,
                                                                                        time_end,
                                                                                        inv_bid,
                                                                                        inv_ask, 
                                                                                        max_events_intra, 
                                                                                        aes
                                                                                        )
    
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds and generated {len(times)} events.")

    times = times.astype(np.float64, copy=False)
    states = np.ascontiguousarray(states)
    print('2')

    df = to_dataframe_short(times, p_mids, sides, events, states, K)
    print('3')

    df.to_parquet(
        "simulations/lob_permanent.parquet",
        engine="pyarrow",
        compression="zstd",    
        index=False
    )
    print('4')