import numpy as np
from numba import njit

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