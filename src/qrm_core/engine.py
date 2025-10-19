import numpy as np
from numba import njit
from math import isnan

from .sampling import choose_next_event_min, update_LOB

"""  
    Simulate the QRM model on [0, time_end] using Numba. All the events are stored in arrays.
    The function returns the arrays of: 
        - times
        - mid prices
        - reference prices
        - sides (1: bid, 2: ask)
        - depths (1 to K) depth of the event (wrt the reference price)
        - events (1: limit, 2: cancel, 3: market, 4: trader)
        - redrawns (0: False, 1: True)
        - states: volumes q_i with format [q_bid1, q_bid2, ..., q_bidK, q_ask1, q_ask2, ..., q_askK]

    NB:
        - rate_int_all: intensities. Shape (2, K, Q+1, 3) for (side, depth, queue size, event type). 
        - time_end: end time of the simulation. 
        - inv_bid, inv_ask: invariant distributions for bid and ask sides. Shape (K, Q+1). 
        - max_events_intra: maximum number of events to simulate (preallocation). 
        - aes: average event size (calibrated on data). Shape (K,). 
"""

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
                     max_events_intra: int, 
                     aes: np.ndarray, 
                     max_nb_events=np.nan
                    ):
    

    K, Q1 = rate_int_all.shape[1:3]
    Q = Q1 - 1
    t = time
    p_mid_old = p_mid
    count = 0

    times   = np.empty(max_events_intra, np.float64)
    p_mids  = np.empty(max_events_intra, np.float64)
    p_refs  = np.empty(max_events_intra, np.float64)
    sides   = np.empty(max_events_intra, np.int8)
    depths  = np.empty(max_events_intra, np.int8)
    events  = np.empty(max_events_intra, np.int8)
    redrawns = np.empty(max_events_intra, np.int8)
    states  = np.empty((max_events_intra, 2*K), np.int8)

    while True:
        # build rates
        n_rates = 2 * K * 3
        rates   = np.empty((n_rates, 4), np.float64)
        idx     = 0
        for side in range(2):
            for d in range(K):
                qsz = state[side * K + d]
                for ev_type in range(3):
                    r = rate_int_all[side, d, qsz, ev_type]
                    rates[idx, 0] = r
                    rates[idx, 1] = side + 1
                    rates[idx, 2] = d + 1
                    rates[idx, 3] = ev_type + 1
                    idx += 1

        # generate next LOB event
        nb, na, st2, sf, dp, ef, t = choose_next_event_min(K, Q, rates, state, t)
        if (isnan(max_nb_events)) and (t > time_end):
            break
        elif (not isnan(max_nb_events)) and (count >= 1*max_nb_events):
            break
        
        # mid‐price update
        new_pmid = 0.5 * ((p_ref + tick * (na + 0.5)) + (p_ref - tick * (nb + 0.5)))
        Δp_mid = abs(new_pmid - p_mid_old)
        Δp_mid_bool = (Δp_mid > tick / 10)
        
        # assess if a limit order arrived inside the bid-ask spread at side s ({-1,1}) and Q_{-s} is not empty
        # in this case, the QRM does not allow a reference price move
        opp_queue_empty = True
        if (ef == 1) and ((sf == 1 and Δp_mid_bool and na == 0) or (sf == 2 and Δp_mid_bool and nb == 0)):
            opp_queue_empty = False

        state = st2
        redrawn = 0
        # update the reference price 
        if Δp_mid_bool and opp_queue_empty:
            p_mid_move = 1 if new_pmid > p_mid_old else -1

            new_pmid, p_ref, state, redrawn = update_LOB(
                                K, p_ref, state, p_mid_move,
                                theta, theta_reinit, tick,
                                inv_bid, inv_ask, aes)

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