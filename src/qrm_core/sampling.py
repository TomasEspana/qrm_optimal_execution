import numpy as np
from numba import njit


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