import numpy as np
import pandas as pd
from numba import njit


class IntensityTable:
    """
        Stores order-flow intensities λ^type(depth, queue_size).
        See Section 2.3.1 of Huang et al. (2015).
    """
    def __init__(
        self,
        max_depth: int,
        max_queue: int,
        types = ("limit", "cancel", "market"),
    ):
        
        self.max_depth = max_depth # also denoted by K in the paper
        self.max_queue = max_queue # max queue size of LOB (in units of AES)
        self.types = types         # types of order flow (limit, cancel, market)
        self._data = np.zeros((max_depth, max_queue + 1, 2, len(types))) # (depth, queue_size, type)
        self._type_index = {t: i for i, t in enumerate(types)} # type -> index
        self._side_index = {side: i for i, side in enumerate(['bid', 'ask'])} # side -> index


    def set_intensity(
        self,
        depth: int,       
        queue: int,       
        side: float,      
        type_name: str,  
        value: float
    ):
        """
            Sets intensity for given depth, queue size, side and type.
        """
        d = min(depth, self.max_depth) - 1
        q = min(queue, self.max_queue)
        si = self._side_index[side]
        ti = self._type_index[type_name] 
        self._data[d, q, si, ti] = value

    def get_intensities(
        self,
        depth: int,
        queue: int,
        side: float
    ):
        """
            Returns intensities for all types at given depth, queue size and side.
        """
        d = min(depth, self.max_depth) - 1
        q = min(queue, self.max_queue)
        si = self._side_index[side]
        raw = self._data[d, q, si, :]
        return {t: raw[self._type_index[t]] for t in self.types}
    

def compute_invariant_distribution(
        side: float,
        intensity_table: IntensityTable,
        dump_path: str
    ):
    """
        Precompute and store the invariant distributions $π_i(n)$ for each queue depth $i$ in an array.
        Refer to Section 2.3.3 of Huang et al. (2015).

    Inputs:
        - side: 'bid', 'ask' or None. We allow bid-ask asymmetry and thus two different invariant distributions.
        - intensity_table: IntensityTable object with order-flow intensities.
        - dump_path: path to .npy file to save the invariant distribution.
    
    Outputs:
        - None. The invariant distribution is saved in a .npy file.
    """
    if side not in [None, 'bid', 'ask']:
        raise ValueError("side must be 'bid', 'ask' or None")
    elif side in [None, 'bid']:
        intensities = intensity_table._data[:, :, 0, :]
    elif side == 'ask':
        intensities = intensity_table._data[:, :, 1, :]

    K, Q_plus_1, *_ = intensities.shape           # max depth, max queue size + 1, number of types
    Q = Q_plus_1 - 1                              # max queue size (in units of AES)
    type_to_index = intensity_table._type_index   # type -> index
    all_pi = np.zeros((K, Q + 1))                 # invariant distribution for each depth (bid-ask symmetry)

    for i in range(K):
        # Arrival/departure ration vector $\rho_i$
        lamL = intensities[i][:-1, type_to_index['limit']]
        lamC = intensities[i][1:, type_to_index['cancel']]
        lamM = intensities[i][1:, type_to_index['market']]
        rho =  lamL / (lamC + lamM)

        pi = np.zeros(Q+1)
        pi_0 = 1 / (1 + np.sum(np.cumprod(rho)))
        pi[0] = pi_0
        pi[1:] = pi_0 * np.cumprod(rho)
        print('sum pi not normalized', np.sum(pi)) # not exactly 1 as we limit the queue size to maximum Q
        pi /= np.sum(pi)                           # normalize
        all_pi[i] = pi

    folder_path = 'calibration_data/invariant_distribution/'
    if side in ['bid', 'ask']:
        file_path = dump_path[:-4] + '_' + side + dump_path[-4:]
        np.save(folder_path + file_path, all_pi)
    else:
        np.save(folder_path + dump_path, all_pi)


# -----------------------------------------------------------------------------
# 1) Draw from invariant distribution
# -----------------------------------------------------------------------------
@njit
def sample_stationary_lob(inv_dist: np.ndarray, depths: np.ndarray):
    K, Q1 = inv_dist.shape
    # full‐LOB sampling
    if depths.size == 0:
        while True:
            state = np.empty(K, np.int32)
            for i in range(K):
                u = np.random.random()
                cum = 0.0
                for q in range(Q1):
                    cum += inv_dist[i, q]
                    if u < cum:
                        state[i] = q
                        break
            if state.sum() > 0:
                return state
    # partial sampling
    out = np.empty(depths.size, np.int32)
    for j in range(depths.size):
        i = depths[j] - 1
        u = np.random.random()
        cum = 0.0
        for q in range(Q1):
            cum += inv_dist[i, q]
            if u < cum:
                out[j] = q
                break
    return out

# -----------------------------------------------------------------------------
# 2) Pick next event
# -----------------------------------------------------------------------------
@njit
def choose_next_event(K: int,
                      Q: int,
                      total_rate: float,
                      rates: np.ndarray,
                      state: np.ndarray):
    
    n = rates.shape[0]

    while True:
        u = np.random.random()
        cum = 0.0
        idx = 0
        for i in range(n):
            cum += rates[i, 0]
            if u * total_rate < cum:
                idx = i
                break

        side_f = np.int32(rates[idx, 1])
        depth  = np.int32(rates[idx, 2])
        evf    = np.int32(rates[idx, 3])
        pos    = (depth - 1) if side_f == 1 else (K + depth - 1)

        new_state = state.copy()
        skip = False
        if evf == 1:  # limit
            if new_state[pos] < Q:
                new_state[pos] += 1
            else:
                skip = True
        else:         # cancel/market
            if new_state[pos] > 0:
                new_state[pos] -= 1
            else:
                skip = True

        # ensure non‐empty book
        best_bid = -1
        for i in range(K):
            if new_state[i] > 0:
                best_bid = i
                break
        best_ask = -1
        for i in range(K):
            if new_state[K + i] > 0:
                best_ask = i
                break

        if best_bid >= 0 and best_ask >= 0:
            return best_bid, best_ask, new_state, side_f, depth, evf, skip

# -----------------------------------------------------------------------------
# 3) Update on mid‐price move, with separate bid/ask invariants
# -----------------------------------------------------------------------------
@njit
def update_LOB(K: int,
               p_ref: float,
               state: np.ndarray,
               mid_move: int,
               theta: float,
               theta_reinit: float,
               tick: float,
               inv_bid: np.ndarray,
               inv_ask: np.ndarray
               ):
    
    while True:

        new_state = state.copy()
        new_pref  = p_ref
        redrawn   = 0

        if np.random.random() < theta:
            new_pref += tick * mid_move
            old = state.copy()

            if np.random.random() > theta_reinit:
                # SHIFT
                if mid_move < 0: # price went down
                    # update ask
                    new_state[K] = 0
                    new_state[K+1:] = old[K:2*K-1]
                    # update bid
                    new_state[:K-1] = old[1:K]
                    depths = np.empty((1,), np.int32); depths[0] = K
                    samp = sample_stationary_lob(inv_bid, depths)
                    new_state[K - 1] = samp[0]
                else: # price went up
                    # update bid
                    new_state[0] = 0
                    new_state[1:K] = old[:K-1]
                    # update ask
                    new_state[K:2*K-1] = old[K+1:]
                    depths = np.empty((1,), np.int32); depths[0] = K
                    samp = sample_stationary_lob(inv_ask, depths)
                    new_state[2*K - 1] = samp[0]
            else:
                # REDRAW
                redrawn = 1
                new_state[:K] = sample_stationary_lob(inv_bid, np.empty((0,), np.int32))
                new_state[K:] = sample_stationary_lob(inv_ask, np.empty((0,), np.int32))

        # ensure non‐empty book
        best_bid = -1
        for i in range(K):
            if new_state[i] > 0:
                best_bid = i
                break
        best_ask = -1
        for i in range(K):
            if new_state[K + i] > 0:
                best_ask = i
                break

        if best_bid >= 0 and best_ask >= 0:
            p_mid = 0.5 * ((new_pref + tick * (best_ask + 0.5)) +  
                            (new_pref - tick * (best_bid + 0.5)))
            return p_mid, new_pref, new_state, redrawn

# -----------------------------------------------------------------------------
# 4) JIT‐compiled core
# -----------------------------------------------------------------------------

@njit
def simulate_QRM_jit(time: float,
                         p_mid: float,
                         p_ref: float,
                         state: np.ndarray,
                         rate_int_all: np.ndarray,
                         tick: float,
                         theta: float,
                         theta_reinit: float,
                         time_end: float,
                         inv_bid: np.ndarray,
                         inv_ask: np.ndarray, 
                         max_events=200 
                         ):
    
    K, Q_plus_1 = rate_int_all.shape[1:3]
    Q = Q_plus_1 - 1
    t = time
    p_mid_old = p_mid
    count = 0

    times   = np.empty(max_events, np.float64)
    p_mids  = np.empty(max_events, np.float64)
    p_refs  = np.empty(max_events, np.float64)
    sides   = np.empty(max_events, np.int32)
    depths  = np.empty(max_events, np.int32)
    events  = np.empty(max_events, np.int32)
    redrawns = np.empty(max_events, np.int32)
    states  = np.empty((max_events, 2*K), np.int32)

    while True:
        # build rates
        n_rates = 2 * K * 3
        rates   = np.empty((n_rates, 4), np.float64)
        idx     = 0
        total   = 0.0
        for s in range(2):
            for d in range(K):
                qsz = state[s * K + d]
                for e in range(3):
                    r = rate_int_all[s, d, qsz, e]
                    rates[idx, 0] = r
                    rates[idx, 1] = s + 1
                    rates[idx, 2] = d + 1
                    rates[idx, 3] = e + 1
                    total += r
                    idx += 1

        # sample dt
        dt = np.random.exponential(1.0 / total)
        t += dt
        if t > time_end:
            break
    
        nb, na, st2, sf, dp, ef, skip = choose_next_event(
            K, Q, total, rates, state
        )
        if skip:
            continue
        state = st2

        # mid‐price update
        new_pmid = 0.5 * ((p_ref + tick * (na + 0.5)) +
                       (p_ref - tick * (nb + 0.5)))
        redrawn = 0
        if abs(new_pmid - p_mid_old) > 1e-6:
            mid_move = 1 if new_pmid > p_mid_old else -1

            new_pmid, p_ref, state, redrawn = update_LOB(
                                K, p_ref, state, mid_move,
                                theta, theta_reinit, tick,
                                inv_bid, inv_ask)

        times[count] = t
        p_mids[count] = round(2*new_pmid/tick) * tick / 2
        p_refs[count] = round(2*p_ref/tick) * tick / 2
        sides[count] = sf
        depths[count] = dp
        events[count] = ef
        redrawns[count] = redrawn
        states[count] = state.copy()

        p_mid_old = new_pmid
        count += 1

    return (times[:count], p_mids[:count], p_refs[:count],
            sides[:count], depths[:count], events[:count],
            redrawns[:count], states[:count])

# -----------------------------------------------------------------------------
# 5) Python wrapper
# -----------------------------------------------------------------------------
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




def simulate_QRM_env(current_LOB: pd.DataFrame,
                     intensity_table: np.ndarray,
                     tick: float,
                     theta: float,
                     theta_reinit: float,
                     time_end: float,
                     inv_bid: np.ndarray,
                     inv_ask: np.ndarray
                     ):
    # unpack history
    time  = current_LOB['time'].to_numpy()[-1]
    p_mid = current_LOB['p_mid'].to_numpy()[-1]
    p_ref = current_LOB['p_ref'].to_numpy()[-1]

    K = intensity_table.shape[1]
    cols = [f"q_bid{i+1}" for i in range(K)] + [f"q_ask{i+1}" for i in range(K)]
    state = current_LOB[cols].to_numpy()[-1].astype(int)

    (times2, mids2, prefs2,
    sides2, depths2, evs2,
    red2, states2) = simulate_QRM_jit(
        time, p_mid, p_ref, state,
        intensity_table, tick, theta, theta_reinit,
        time_end, inv_bid, inv_ask
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
