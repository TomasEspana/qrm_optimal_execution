import numpy as np 
import time

from src.qrm_core.engine import simulate_QRM_jit
from src.qrm_core.sampling import sample_stationary_lob
from src.qrm_core.intensity import IntensityTable


def to_dataframe(times, p_mids, p_refs, sides, depths, events, redrawn, states, K):
        """
            Convert the LOB to a Pandas DataFrame.
        """
        import pandas as pd

        df = pd.DataFrame({
            'time':   times,
            'p_mid':  p_mids,
            'p_ref':  p_refs,
            'side':   sides,
            'depth':  depths,
            'event':  events,
            'redrawn':redrawn
        })

        df['side']  = df['side'].map({1:'bid', 2:'ask'})
        df['event'] = df['event'].map({1:'limit', 2:'cancel', 3:'market', 4:'trader'})
        df['redrawn'] = df['redrawn'].astype(bool)

        bids = [f"q_bid{i+1}" for i in range(K)][::-1]
        asks = [f"q_ask{i+1}" for i in range(K)]
        cols = bids + asks
        block = states
        for j, name in enumerate(cols):
            if j < K:
                df[name] = block[:, K - j - 1]
            else:
                df[name] = block[:, j]

        return df



if __name__ == "__main__":

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
    one_spread = True
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
    theta = 0.7
    theta_reinit = 0.85
    time_end = 10000
    max_events_intra = 30 * int(time_end) 

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
    dt = time.time() - start
    print(f"Elapsed: {dt:.6f} s")

    df = to_dataframe(times, p_mids, p_refs, sides, depths, events, redrawns, states, K)
    df.to_csv('simulations/test_simulation.csv', index=False)