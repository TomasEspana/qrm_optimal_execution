import numpy as np
from qrm_core.sampling import sample_stationary_lob
from qrm_core.engine import simulate_QRM_jit


class QueueReactiveMarketSimulator:

    def __init__(
        self,
        intensity_table: np.ndarray,
        theta: float,
        theta_reinit: float,
        initial_price: float,
        tick: float,
        inv_bid: np.ndarray,
        inv_ask: np.ndarray,
        trader_times: np.ndarray,
        max_events: int, 
        max_events_intra: int, 
        aes: np.ndarray
    ):
        self.intensity_table = intensity_table
        self.K = intensity_table.shape[1]
        self.theta = theta
        self.theta_reinit = theta_reinit
        self.tick = tick
        self.inv_bid = inv_bid
        self.inv_ask = inv_ask
        self.trader_times = trader_times
        self.initial_price = initial_price
        self.aes = aes

        # logging buffer capacity (pre-allocation)
        self.max_events = max_events             # max number of LOB events to log for one episode
        self.max_events_intra = max_events_intra # max number of events between two trader times
        
        self.step = 0                 # current number of logged events
        self.next_trader_time_idx = 0 # index of the next trader time to process

    def initialize(self):
        """
            Draw initial LOB and log it as event 0.
        """
        self.step = 0
        self.next_trader_time_idx = 0

        self.times   = np.empty(self.max_events, np.float64)
        self.p_mids  = np.empty(self.max_events, np.float64)
        self.p_refs  = np.empty(self.max_events, np.float64)
        self.sides   = np.zeros(self.max_events, np.int8)
        self.depths  = np.zeros(self.max_events, np.int8)
        self.events  = np.zeros(self.max_events, np.int8)
        self.redrawn = np.zeros(self.max_events, np.int8) 
        self.states  = np.empty((self.max_events, 2*self.K), np.int8)

        # sample from invariant distribution
        lob0 = np.empty(2*self.K, np.int8)
        one_spread = False
        
        if one_spread: # force one tick spread
            while True:
                lob0[:self.K]   = sample_stationary_lob(self.inv_bid, np.empty((0,), np.int8))
                lob0[self.K:] = sample_stationary_lob(self.inv_ask, np.empty((0,), np.int8))
                bid_idx = next((i for i in range(self.K) if lob0[i]>0), None)
                nz = np.flatnonzero(lob0[self.K:])
                ask_idx = self.K + nz[0] if len(nz) > 0 else None
                ask_idx_second = self.K + nz[1] if len(nz) > 1 else None
                if bid_idx == 0 and ask_idx == self.K and ask_idx_second == (self.K+1):
                    break
        
        else:
            lob0[:self.K] = sample_stationary_lob(self.inv_bid, np.empty((0,), np.int8))
            lob0[self.K:] = sample_stationary_lob(self.inv_ask, np.empty((0,), np.int8))

        # identify best bid and ask
        bid_idx = next((i for i in range(self.K) if lob0[i]>0), None)
        ask_idx = next((i for i in range(self.K, 2*self.K) if lob0[i]>0), None)
        if bid_idx is None or ask_idx is None:
            raise ValueError("Sampled empty LOB")
        p_mid = 0.5 * ((self.initial_price + self.tick * (ask_idx - self.K + 0.5)) +
                          (self.initial_price - self.tick * (bid_idx + 0.5)))
        p_ref = self.initial_price

        # log the initial state to the LOB
        self._write_batch(
            times=[0.0],
            p_mids=[p_mid],
            p_refs=[p_ref],
            sides=[0],
            depths=[0],
            events=[0],
            redrawns=[False],
            lob_states=[lob0]
        )

    def _write_batch(self, times, p_mids, p_refs,
                     sides, depths, events, redrawns, lob_states):
        """
            Log series of events in the LOB.
        """
        n = len(times)
        if n == 0:
            return
        
        i0 = self.step
        i1 = i0 + n
        self.step = i1
        if i1 > self.max_events:
            raise ValueError(f"Exceeded max_events={self.max_events}")

        self.times  [i0:i1] = times
        self.p_mids [i0:i1] = p_mids
        self.p_refs [i0:i1] = p_refs
        self.sides  [i0:i1] = sides
        self.depths [i0:i1] = depths
        self.events [i0:i1] = events
        self.redrawn[i0:i1] = redrawns
        self.states[i0:i1, :] = np.vstack(lob_states)


    def current_time(self):
        return self.times[self.step - 1]

    def current_mid_price(self):
        return self.p_mids[self.step - 1]

    def current_ref_price(self):
        return self.p_refs[self.step - 1]

    def current_state(self, history_size=1):
        return self.states[self.step - history_size:self.step].copy()[::-1]
    

    def simulate_step(self):
        """
            Run the QRM simulation to the next trader time.
        """
        self.next_trader_time_idx += 1
        next_t = self.trader_times[self.next_trader_time_idx]

        (times, p_mids, p_refs,
         sides, depths, events,
         redrawns, lob_states) = simulate_QRM_jit(
            self.current_time(),
            self.current_mid_price(),
            self.current_ref_price(),
            self.current_state()[0],
            self.intensity_table,
            self.tick, self.theta, self.theta_reinit,
            next_t, self.inv_bid, self.inv_ask, self.max_events_intra, self.aes, 
            np.nan # change this arg. to an integer if you want the trader to act on a number-of-event-frequency (instead of default time-frequency)
        )

        self._write_batch(
            times, p_mids, p_refs,
            sides, depths, events,
            redrawns, lob_states
        )
    

    def close(self):
        import gc

        for attr in ("times", "p_mids", "p_refs",
                     "sides", "depths", "events",
                     "redrawn", "states",
                     "intensity_table", "inv_bid", "inv_ask", "trader_times"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

        gc.collect()