import numpy as np
from numba import njit


""" 
    Auxiliary functions for simulating the QRM model.
    1) `sample_stationary_lob`: redraw the queues from the invariant distribution.
    2) `choose_next_event`: sample the next event in the LOB.
    3) `update_LOB`: update the LOB according to the QRM model.

    `choose_next_event` and `update_LOB` are used in the `simulate_QRM_jit` (.engine.py) function.

"""


# -----------------------------------------------------------------------------
# 1) Sample from invariant distribution
# -----------------------------------------------------------------------------
@njit
def sample_stationary_lob(inv_dist: np.ndarray, depths: np.ndarray):
    """
        Redraw the volumes from the invariant distribution (see Section 3.1 of Huang et al. (2015)). 
    """
    K, Q1 = inv_dist.shape

    # full‐LOB sampling
    if depths.size == 0:
        while True:
            state = np.empty(K, np.int8)
            for i in range(K):
                state[i] = np.random.choice(np.arange(Q1), p=inv_dist[i])
            if state.sum() > 0:
                return state
            
    # partial sampling
    out = np.empty(depths.size, np.int8)
    for depth in depths:
        d = depth - 1
        out[depth] = np.random.choice(np.arange(Q1), p=inv_dist[d])

    return out


# -----------------------------------------------------------------------------
# 2) Sample next event
# -----------------------------------------------------------------------------
@njit
def choose_next_event(K: int,
                      Q: int,
                      total_rate: float,
                      rates: np.ndarray,
                      state: np.ndarray):
    """
        The sampling of the next LOB events is done by summing the rates conditional on the current state.
        Note that we use 'choose_next_event_bis' instead in the QRM simulator.
        The difference is just the sampling method (summing vs. minimum), which are equivalent but we prefer to use 'choose_next_event_bis'.
    """
    
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

        side_f = np.int8(rates[idx, 1])
        depth  = np.int8(rates[idx, 2])
        evf    = np.int8(rates[idx, 3])
        pos    = (depth - 1) if side_f == 1 else (K + depth - 1)

        new_state = state.copy()
        skip = False
        if evf == 1:  # limit
            if new_state[pos] < Q:
                new_state[pos] += 1
            else:
                print('FORBIDDEN: limit order exceeds max queue size \n')
                print('DEPTH', depth, 'SIDE', side_f)
                skip = True
        else:         # cancel/market
            if new_state[pos] > 0:
                new_state[pos] -= 1
            else:
                print('FORBIDDEN: cancel/market order on empty queue \n')
                print('DEPTH', depth, 'SIDE', side_f)
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
        
        count += 1
        if best_bid >= 0 and best_ask >= 0:
            return best_bid, best_ask, new_state, side_f, depth, evf, skip
        

@njit
def choose_next_event_bis(K: int,
                      Q: int,
                      rates: np.ndarray,
                      state: np.ndarray, 
                      t: float
                      ):
    
    """ 
        The sampling of the next LOB events is done by picking the minimum time among the rates conditional on the current state.
        The sampling (summing rates or minimum) is the only difference between this function and `choose_next_event`.
        We pick 'choose_next_event_bis'.
    """
    
    n = rates.shape[0]
    rates_array = rates[:, 0]
    dt_array = np.empty(n)

    while True:
        # sample dt
        for i in range(n):
            if rates_array[i] == 0:
                dt_array[i] = np.inf
            else:
                dt_array[i] = np.random.exponential(1.0 / rates_array[i])
        idx = np.argmin(dt_array)
        t += dt_array[idx]

        side_f = np.int8(rates[idx, 1])
        depth  = np.int8(rates[idx, 2])
        evf    = np.int8(rates[idx, 3])
        pos    = (depth - 1) if side_f == 1 else (K + depth - 1)

        new_state = state.copy()
        skip = False
        if evf == 1:  # limit
            if new_state[pos] < Q:
                new_state[pos] += 1
            else:
                # print('FORBIDDEN: limit order exceeds max queue size \n')
                # print('DEPTH', depth, 'SIDE', side_f)
                skip = True
        else:         # cancel/market
            if new_state[pos] > 0:
                new_state[pos] -= 1
            else:
                # print('FORBIDDEN: cancel/market order on empty queue \n')
                # print('DEPTH', depth, 'SIDE', side_f)
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
        
        if (best_bid >= 0 and best_ask >= 0) and not skip:
            return best_bid, best_ask, new_state, side_f, depth, evf, t


# -----------------------------------------------------------------------------
# 3) Update LOB
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
               inv_ask: np.ndarray,
               aes: np.ndarray
               ):
    
    Q = inv_bid.shape[1] - 1
    while True:

        new_state = state.copy()
        new_pref  = p_ref
        redrawn   = 0

        if np.random.random() < theta: # p_ref changes
            new_pref += tick * mid_move
            old = state.copy()

            if np.random.random() > theta_reinit:
                # Shift the queues
                if mid_move < 0: # price went down
                    # update ask
                    new_state[K] = 0
                    new_state[K+1:] = np.minimum(Q, np.rint(
                            old[K:2*K-1] * np.array([aes[i] / aes[i+1] for i in range(K-1)])
                        )).astype(np.int8)
                    # update bid
                    new_state[:K-1] = np.minimum(Q, np.rint(
                            old[1:K] * np.array([aes[i+1] / aes[i] for i in range(K-1)])
                        )).astype(np.int8)
                    depths = np.empty((1,), np.int8); depths[0] = K
                    samp = sample_stationary_lob(inv_bid, depths)
                    new_state[K - 1] = samp[0]

                else: # price went up
                    # update bid
                    new_state[0] = 0
                    new_state[1:K] = np.minimum(Q, np.rint(
                            old[:K-1] * np.array([aes[i] / aes[i+1] for i in range(K-1)])
                        )).astype(np.int8)
                    # update ask
                    new_state[K:2*K-1] = np.minimum(Q, np.rint(
                            old[K+1:] * np.array([aes[i+1] / aes[i] for i in range(K-1)])
                        )).astype(np.int8)
                    depths = np.empty((1,), np.int8); depths[0] = K
                    samp = sample_stationary_lob(inv_ask, depths)
                    new_state[2*K - 1] = samp[0]
            else:
                # REDRAW
                redrawn = 1
                new_state[:K] = sample_stationary_lob(inv_bid, np.empty((0,), np.int8))
                new_state[K:] = sample_stationary_lob(inv_ask, np.empty((0,), np.int8))

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