import numpy as np
import pandas as pd
from typing import Tuple, Dict



# IMPROVEMENTS STILL PENDENT:
    # - tracking and storing of dictionarys that seem useless in simulate_model_I and simulate_QRM


class IntensityTable:
    """
        Stores order-flow intensities λ^type(depth, queue_size).
        See Section 2.3.1 of Huang et al. (2015).
    """
    def __init__(
        self,
        max_depth: int,
        max_queue: int,
        types: Tuple[str, ...] = ("limit", "cancel", "market"),
    ):
        
        self.max_depth = max_depth # also denoted by K in the paper
        self.max_queue = max_queue # max queue size of LOB (in units of AES)
        self.types = types         # types of order flow (limit, cancel, market)
        self._data = np.zeros((max_depth, max_queue + 1, 2, len(types))) # (depth, queue_size, type)
        self._type_index: Dict[str, int] = {t: i for i, t in enumerate(types)} # type -> index
        self._side_index: Dict[int, str] = {side: i for i, side in enumerate(['bid', 'ask'])} # side -> index


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



def sample_stationary_lob_from_file(
        side: float,
        distribution_file: str, 
        rng: np.random.Generator, 
        depths: list = None
        ):

    """
        Sample queue sizes from the invariant distribution (Section 2.3.3).

    Inputs:
        - side: 'bid', 'ask' or None. We allow bid-ask asymmetry and thus two different invariant distributions.
        - distribution_file: path to .npy file with invariant distribution with shape (K, Q+1)
        - rng: np.random.Generator for reproducibility
        - depths: optional list of depths to sample (1-indexed). If None, sample the entire LOB.

    Outputs:
        - If depths is None: array of shape (2K,) → [q_bid1,…,q_bidK, q_ask1,…,q_askK]
        - Else: array of shape (len(depths),) with sampled queue sizes targeting the given depths.
    """
    if side not in [None, 'bid', 'ask']:
        raise ValueError("side must be 'bid', 'ask' or None")
    elif side in ['bid', 'ask']:
        path = distribution_file[:-4] + '_' + side + distribution_file[-4:]
    else:
        path = distribution_file

    all_pi = np.load(path)
    K, Q_plus_1 = all_pi.shape

    if depths is None:
        # FULL LOB SAMPLING
        while True:
            state = np.zeros(K, dtype=int)
            for i in range(K):
                probs = all_pi[i]
                level = rng.choice(np.arange(Q_plus_1), p=probs)
                state[i] = level
            if np.sum(state) == 0:
                # If all bids or asks are zero (empty order book), resample
                continue
            else:
                return state

    else:
        # PARTIAL SAMPLING
        sampled = []
        for d in depths:
            if not (1 <= d <= K):
                raise ValueError(f"Depth {d} out of bounds: must be in 1,…,{K}")
            probs = all_pi[d - 1]
            sampled.append(rng.choice(np.arange(Q_plus_1), p=probs))

        return np.array(sampled, dtype=int)


def choose_next_event(K, Q, total_rate, rng, rates, state):
    """
        Choose the next event based on the rates and the current state of the LOB.
        Make sure there will always be a bid and an ask in the LOB after the event (security for RL training).
    
    Inputs:
        - K, Q: max depth and queue size of LOB
        - total_rate: sum of all the intensities given the current state of the LOB
        - rng: np.random.Generator for reproducibility
        - rates: list of tuples (intensity, side, depth, event) for each event type
        - state: current state of the LOB in the format [q_bid1, ..., q_bidK, q_ask1, ..., q_askK] (in AES units)
    
    Outputs:
        - new_best_bid: index of the best bid in the new state
        - new_best_ask: index of the best ask in the new state
        - new_state: updated state of the LOB after the event
        - chosen: tuple (side, depth, event) of the chosen (sampled) event
        - skip: boolean indicating if the event was a no-op (e.g. cancel/market order at an empty queue or limit order at full queue)
    """

    while True:  # Keep trying until a valid event is found (i.e., both bid and ask are non-empty)
        try:
            new_state = state.copy()

            # Choose the event type, side and depth
            probs = np.array([rate for rate, *_ in rates]) / total_rate
            evt_idx = rng.choice(np.arange(len(rates)), p=probs)
            _, side, depth, ev = rates[evt_idx]
            chosen = (side, depth, ev)
            idx = (depth - 1) if side == 'bid' else (depth - 1 + K)

            skip = False 
            if ev == 'limit':
                if new_state[idx] == Q:
                    skip = True
                else:
                    new_state[idx] += 1
            else:
                if new_state[idx] == 0:
                    skip = True
                else:
                    new_state[idx] -= 1

            new_best_bid = next((i for i in range(K) if new_state[i] > 0), None)
            new_best_ask = next((i for i in range(K) if new_state[K + i] > 0), None)

            if new_best_bid is not None and new_best_ask is not None:
                return new_best_bid, new_best_ask, new_state, chosen, skip
            else:
                raise ValueError("No best bid or ask found in the state.")
            
        except ValueError:
            continue  # Try again if best bid/ask not found



def update_LOB(K, ref_price, state, mid_move, rng, theta, theta_reinit, tick, inv_dist_file, bid_ask_sym):
    """
        Update LOB after mid-price change according to Section 3.1.1. 
    
    Inputs:
        - K: max depth of LOB
        - ref_price: current reference price
        - state: current state of the LOB in the format [q_bid1, ..., q_bidK, q_ask1, ..., q_askK] (in AES units)
        - mid_move: 1 if mid price went up, -1 if down
        - rng: np.random.Generator for reproducibility
        - theta: probability of p_ref change after p_mid change (Section 3.1.1.)
        - theta_reinit: probability of redrawing LOB from invariant distribution after p_ref change (Section 3.1.1.)
        - tick: tick size
        - inv_dist_file: path to .npy file with invariant distribution
        - bid_ask_sym: boolean indicating if intensities are bid-ask symmetric
    
    Outputs:
        - p_mid: new mid price
        - p_ref: new reference price
        - new_state: updated state of the LOB after the event
        - redrawn: boolean indicating if the LOB was redrawn from invariant distribution
    """
    
    while True:

        p_ref = ref_price
        try:
            redrawn = False
            change_p_ref = rng.random() # probability of p_ref change after p_mid change
            ext_info = rng.random()     # probability of redrawing LOB from invariant distribution after p_ref change
            new_state = state.copy()
            
            if change_p_ref < theta:
                p_ref += tick * mid_move
                old_state = state.copy()
                
                # No external information, we shift the queue sizes
                if ext_info > theta_reinit: 
                    if mid_move < 0: # price went down
                        # update ask 
                        new_state[K:K+1] = 0
                        new_state[K+1:] = old_state[K:2*K-1]
                        # update bid
                        new_state[:K-1] = old_state[1:K]
                        if bid_ask_sym:
                            new_state[K-1:K] = sample_stationary_lob_from_file(None, inv_dist_file, rng, depths=[K])
                        else:
                            new_state[K-1:K] = sample_stationary_lob_from_file('bid', inv_dist_file, rng, depths=[K])

                    else: # price went up
                        # update bid
                        new_state[:1] = 0
                        new_state[1:K] = old_state[:K-1]
                        # update ask 
                        new_state[K:2*K-1] = old_state[K+1:]
                        if bid_ask_sym:
                            new_state[2*K-1:] = sample_stationary_lob_from_file(None, inv_dist_file, rng, depths=[K])
                        else:
                            new_state[2*K-1:] = sample_stationary_lob_from_file('ask', inv_dist_file, rng, depths=[K])
                
                else: # external information: redraw from invariant distribution (Section 3.1.1)
                    redrawn = True
                    new_state = np.zeros(2*K, dtype=int)
                    if bid_ask_sym:
                        new_state[:K] = sample_stationary_lob_from_file(None, inv_dist_file, rng)
                        new_state[K:] = sample_stationary_lob_from_file(None, inv_dist_file, rng)
                    else:
                        new_state[:K] = sample_stationary_lob_from_file('bid', inv_dist_file, rng)
                        new_state[K:] = sample_stationary_lob_from_file('ask', inv_dist_file, rng)

            new_best_bid = next((i for i in range(K) if new_state[i] > 0), None)
            new_best_ask = next((i for i in range(K) if new_state[K + i] > 0), None)

            if new_best_bid is not None and new_best_ask is not None:
                p_mid = 0.5 * ((p_ref + tick*(new_best_ask+0.5)) + (p_ref - tick*(new_best_bid+0.5)))
                return p_mid, p_ref, new_state, redrawn
            
            else:
                raise ValueError("No best bid or ask found in the state.")
            
        except ValueError:
            continue # Try again if best bid/ask not found
  



def simulate_model_I(
    current_LOB: pd.DataFrame,
    intensity_table: IntensityTable,
    tick: float,
    theta: float,
    theta_reinit: float,
    max_steps: int,
    time_end: float,           
    rng: np.random.Generator,
    inv_dist_file: str,
    bid_ask_sym: bool
    ): 
    """
        Simulation of Model I (Section 2.3.1).
        Simulation stops either on (max_steps or time_end) or on mid-price move.

    Inputs:
        - current_LOB: DataFrame of the LOB.
        - intensity_table: IntensityTable object with order-flow intensities.
        - tick: tick size 
        - theta: probability of p_ref change after p_mid change (Section 3.1.1.)
        - theta_reinit: probability of redrawing LOB from invariant distribution after p_ref change (Section 3.1.1.)
        - max_steps: maximum number of steps to simulate (optional).
        - time_end: end time, in seconds, for simulation (optional).
        - rng: np.random.Generator for reproducibility
        - inv_dist_file: path to .npy file with invariant distribution
        - bid_ask_sym: boolean indicating if intensities are bid-ask symmetric

    Outputs:
        - Dataframe of updated LOB with columns
            ['time', 'p_mid', 'p_ref', 'side', 'depth', 'event', 'queue_size', 'q_bidK', ..., 'q_bid1', 'q_ask1', ..., 'q_askK']
        - mid_termination: boolean indicating if simulation ended because of mid-price move.
        - step: number of new events in the LOB.
    """    
    # Parameters
    Q = intensity_table.max_queue            # max queue size of LOB (in units of AES)
    K = intensity_table.max_depth            # max depth of LOB (in ticks)
    last_row = current_LOB.iloc[-1].copy()   # last event of the LOB
    t = last_row['time']
    step = 0
    records = []                             # record of new events to come in the LOB
    mid_termination = False                  # end simulation because of mid price change

    # Prices
    p_mid_old = last_row['p_mid']
    ref_price = last_row['p_ref']

    # Current state of the LOB in the format [q_bid1, ..., q_bidK, q_ask1, ..., q_askK] (in AES units)
    columns = [f'q_bid{i+1}' for i in range(K)] + [f'q_ask{i+1}' for i in range(K)]
    state = last_row[columns].values.copy().astype(int)


    while True:
        # max steps
        if max_steps is not None and step >= max_steps:
            break
        
        # Intensities for each event type (limit, cancel, market) at each depth (bid/ask)
        rates = []
        for side in ['bid', 'ask']:
            for d in range(1, K+1):
                idx = (d-1) if side=='bid' else (K + d-1)
                qsize = state[idx]
                ints = intensity_table.get_intensities(d, qsize, side)
                rates.append((ints['limit'],  side, d, 'limit'))
                rates.append((ints['cancel'], side, d, 'cancel'))
                rates.append((ints['market'], side, d, 'market'))

        # sum of the intensities for all events (queues are supposed independent in Model I)
        total_rate = sum(r for r, *_ in rates)
        if total_rate <= 0:
            raise ValueError(f"Total_rate <= 0 at step {step}, t={t:.4f}, stopping simulation")

        # Generate next event time
        delta_t = rng.exponential(1/total_rate)
        t += delta_t

        if time_end is not None and t > time_end:
          break

        # caution: we don't want an event that empties the bid or the ask (security for RL training)
        new_best_bid, new_best_ask, state, chosen, skip = choose_next_event(K, Q, total_rate, rng, rates, state)
        if skip: # an order that doesn't change the LOB arrived (e.g. cancel at an empty queue)
            continue

        side, depth, ev = chosen
        p_mid = 0.5 * ((ref_price + tick*(new_best_ask+0.5)) + (ref_price - tick*(new_best_bid+0.5)))

        redrawn = False
        if np.abs(p_mid - p_mid_old) > 1e-6: # mid price change
          mid_termination = True
          mid_move = int(p_mid > p_mid_old) - int(p_mid < p_mid_old) # 1 if mid price went up, -1 if down
        
          # caution: we don't want an update that empties the bid or the ask (security for RL training)
          p_mid, ref_price, state, redrawn = update_LOB(K, ref_price, state, mid_move, 
                                                              rng, theta, theta_reinit, tick, inv_dist_file, bid_ask_sym)

        p_mid_old = p_mid
        # Record event + full LOB snapshot
        rec = {'time': t, 
               'p_mid': round(2*p_mid/tick) * tick / 2, 
               'p_ref': round(2*ref_price/tick) * tick / 2,
               'side': side,
               'depth': depth, 
               'event': ev,
               'redrawn': redrawn
               }

        for j in range(K):
            rec[f'q_bid{j+1}'] = state[j]
            rec[f'q_ask{j+1}'] = state[K+j]

        records.append(rec)
        step += 1

        # stop on mid-price move
        if mid_termination:
            break

      
    if len(records) > 0:
        new_events_LOB = pd.DataFrame(records)
        updated_LOB = pd.concat([current_LOB, new_events_LOB], ignore_index=True)
        return updated_LOB, mid_termination, step
    else:
        return current_LOB, mid_termination, step
    


def simulate_QRM(
    current_LOB: pd.DataFrame,
    intensity_table: IntensityTable,
    tick: float,
    theta: float,
    theta_reinit: float,
    time_end: float,
    rng: np.random.Generator,
    inv_dist_file: str, 
    bid_ask_sym: bool
    ):
    
    """
        Simulation of Model III (Section 3.1.1.) until time_end (in seconds).
        Repeats the following until time_end: 
            - run Model I until the mid-price changes, 
            - update the LOB accordingly.

    Inputs:
        - current_LOB: DataFrame with the LOB up to time t=initial_time.
        - intensity_table: IntensityTable object with order-flow intensities.
        - tick: tick size
        - theta: probability of p_ref change after p_mid change (Section 3.1.1.)
        - theta_reinit: probability of redrawing LOB from invariant distribution after p_ref change (Section 3.1.1.)
        - time_end: end time for simulation.
        - rng: np.random.Generator for reproducibility
        - inv_dist_file: path to .npy file with invariant distribution
        - bid_ask_sym: boolean indicating if intensities are bid-ask symmetric
    
    Outputs:
        - Dataframe of updated LOB with columns 
            ['time', 'p_mid', 'p_ref', 'side', 'depth', 'event', 'queue_size', 'q_bidK', ..., 'q_bid1', 'q_ask1', ..., 'q_askK']
        - nb_events_dic: dictionary with the number of events in each type of simulation.
    """

    nb_events_dic = {'price_change': [], 'time_limit': []} # tracks the number of events in each type of simulation

    LOB = current_LOB.copy()
    step = 0

    ### Simulation of Model I on [initial_time, time_end[
    ### where initial_time is the last time of the current LOB

    # Either the simulation ends on time limit or mid-price move
    # If it ends on price move, we resimulate again till the end of the time interval

    mid_termination = True # simulation ended because of mid price change
    while mid_termination: # resimulate until time limit

        LOB, mid_termination, nb_events = simulate_model_I(
            current_LOB=LOB,
            intensity_table=intensity_table,
            tick=tick,
            theta=theta,
            theta_reinit=theta_reinit,
            max_steps=None,
            time_end=time_end,
            rng=rng,
            inv_dist_file=inv_dist_file, 
            bid_ask_sym=bid_ask_sym
        )
        
        if mid_termination:
            nb_events_dic['price_change'].append((step, nb_events))
        else:
            nb_events_dic['time_limit'].append((step, nb_events))

        step += 1
 
    return LOB, nb_events_dic