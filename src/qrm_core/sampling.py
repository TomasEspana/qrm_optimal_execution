import numpy as np
from numba import njit


""" 
GENERAL DESCRIPTION:

    This file gathers auxiliary functions for simulating the QRM model.
        1) `sample_stationary_lob`: redraw the volumes from the invariant distribution.
        2) `choose_next_event_min`: sample the next event in the LOB.
        3) `update_LOB`: update the LOB after a mid-price change.

    `choose_next_event_min` and `update_LOB` are used in the `simulate_QRM_jit` function (.engine.py).

"""


# -----------------------------------------------------------------------------
# 1) Sample from invariant distribution
# -----------------------------------------------------------------------------

@njit
def sample_stationary_lob(inv_dist: np.ndarray, depths: np.ndarray):
    """
        Redraw the volumes from the invariant distribution (see Section 3.1 of Huang et al. (2015)). 
    
    Input:
        - inv_dist: invariant distribution (shape: K x Q+1)
        - depths: array of depths to sample (shape: D). If empty, sample the full LOB.
    Output:
        - out: sampled volumes (shape: D)
    """
    K, Q1 = inv_dist.shape

    # full‐LOB sampling
    if depths.size == 0:
        while True:
            state = np.empty(K, np.int8)
            for i in range(K):
                u = np.random.random()
                cum = 0.0
                for q in range(Q1):
                    cum += inv_dist[i, q]
                    if u < cum:
                        state[i] = np.int8(q)
                        break
            if state.sum() > 0:
                return state
            
    # partial sampling
    elif depths.size > 1:
        raise NotImplementedError("Partial sampling with multiple depths is not implemented.")
    
    else:
        out = np.empty(1, np.int8)
        d = depths[0] - 1
        u = np.random.random()
        cum = 0.0
        for q in range(Q1):
            cum += inv_dist[d, q]
            if u < cum:
                out[0] = np.int8(q)
                break

    return out


# -----------------------------------------------------------------------------
# 2) Sample next event
# -----------------------------------------------------------------------------

@njit
def choose_next_event_min(K: int,
                          Q: int,
                          rates: np.ndarray,
                          state: np.ndarray, 
                          t: float
                        ):
    
    """ 
        Generate the next LOB event.
    
    Input:
        - K: maximum depth of the LOB
        - Q: maximum queue size at each level
        - rates: array of shape (6*K, 4) with (intensity, side, depth, event type)
        - state: current LOB state (volumes at each level)
        - t: time of the last LOB event

    Output:
        - best_bid, best_ask: indexes of new best bid/ask after the event (0,...,K-1)
        - new_state: volumes [q_bid1, q_bid2,..., q_bidK, q_ask1, q_ask2,..., q_askK]
        - side_f: side of the event (1: bid, 2: ask)
        - depth: depth of the event (1 to K)
        - evf: event type (1: limit, 2: cancel, 3: market)
        - t: time of the new event
    """
    
    n = rates.shape[0]
    rates_array = rates[:, 0]
    dt_array = np.empty(n)
    kk = 0

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
        # limit order
        if evf == 1:
            if new_state[pos] < Q:
                new_state[pos] += 1
            else:
                skip = True
        # cancel/market order
        else:         
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
        
        if (best_bid >= 0 and best_ask >= 0) and not skip:
            return best_bid, best_ask, new_state, side_f, depth, evf, t
        
        kk += 1
        if kk > 10:
            raise ValueError("Unable to sample next event: too many rejections.")
        


@njit
def choose_next_event_deprecated(K: int,
                      Q: int,
                      total_rate: float,
                      rates: np.ndarray,
                      state: np.ndarray):
    """
        The sampling of the next LOB events is done by summing the rates conditional on the current state.
        Note that we use 'choose_next_event_min' instead in the QRM simulator.
        The difference is just the sampling method (summing vs. minimum), which are equivalent but we prefer to use 'choose_next_event_min'.
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
    


# -----------------------------------------------------------------------------
# 3) Update LOB after a mid-price change
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
    """
        Update the LOB after a mid-price change. 

    Input:
        - K: maximum depth of the LOB
        - p_ref: current reference price
        - state: current LOB state (volumes at each level)
        - mid_move: direction of the mid-price move (1: up, -1: down)
        - theta: probability that p_ref changes at a mid-price move
        - theta_reinit: probability of a full redraw after a mid-price move
        - tick: tick size
        - inv_bid, inv_ask: invariant distributions for bid and ask sides (shape: K x Q+1)
        - aes: average event size (calibrated on data). Shape (K,). 

    Output:
        - p_mid: new mid-price
        - new_pref: new reference price
        - new_state: new LOB state (volumes at each level)
        - redrawn: indicator of a full redraw (1: True, 0: False)
    """
    Q = inv_bid.shape[1] - 1
    kk = 0

    while True:
        new_state = state.copy()
        new_pref  = p_ref
        redrawn   = 0

        if np.random.random() < theta: # p_ref changes
            new_pref += tick * mid_move

            if np.random.random() > theta_reinit:
                # Shift the queues
                if mid_move < 0: # price went down
                    # update ask
                    new_state[K] = 0
                    new_state[K+1:] = np.minimum(Q, np.rint(
                            state[K:2*K-1] * np.array([aes[i] / aes[i+1] for i in range(K-1)])
                        )).astype(np.int8)
                    # update bid
                    new_state[:K-1] = np.minimum(Q, np.rint(
                            state[1:K] * np.array([aes[i+1] / aes[i] for i in range(K-1)])
                        )).astype(np.int8)
                    depths = np.empty((1,), np.int8); depths[0] = K
                    samp = sample_stationary_lob(inv_bid, depths)
                    new_state[K - 1] = samp[0]

                else: # price went up
                    # update bid
                    new_state[0] = 0
                    new_state[1:K] = np.minimum(Q, np.rint(
                            state[:K-1] * np.array([aes[i] / aes[i+1] for i in range(K-1)])
                        )).astype(np.int8)
                    # update ask
                    new_state[K:2*K-1] = np.minimum(Q, np.rint(
                            state[K+1:] * np.array([aes[i+1] / aes[i] for i in range(K-1)])
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
        
        kk += 1
        if kk > 10:
            raise ValueError("Unable to update LOB: sampled empty book too many times.")