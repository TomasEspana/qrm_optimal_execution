import numpy as np
import pandas as pd
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
        max_events_intra: int
    ):
        # core parameters
        self.intensity_table = intensity_table
        self.K = intensity_table.shape[1]
        self.theta = theta
        self.theta_reinit = theta_reinit
        self.tick = tick
        self.inv_bid = inv_bid
        self.inv_ask = inv_ask
        self.trader_times = trader_times
        self.initial_price = initial_price

        # logging buffer capacity
        self.max_events = max_events # max number of events to log for one episode
        self.max_events_intra = max_events_intra # max number of events per intra-trader step 
        self.step = 0   # how many events logged so far
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
        self.sides   = np.zeros(self.max_events, np.int8) # 1 = bid, 2 = ask
        self.depths  = np.zeros(self.max_events, np.int8) # depth of the event (nb of executed shares when trader action)
        self.events  = np.zeros(self.max_events, np.int8) # 1 = limit, 2 = cancel, 3 = order, 4 = trader
        self.redrawn = np.zeros(self.max_events, np.int8) # 0 = not redrawn, 1 = redrawn
        self.states  = np.empty((self.max_events, 2*self.K), np.int8) # [q_bid1, ..., q_bidK, q_ask1, ..., q_askK] format

        # sample from invariant distribution
        lob0 = np.empty(2*self.K, np.int8)
        lob0[:self.K]   = sample_stationary_lob(self.inv_bid, np.empty((0,), np.int8))
        lob0[self.K:] = sample_stationary_lob(self.inv_ask, np.empty((0,), np.int8))

        # log the initial state to the LOB
        self._write_batch(
            times=[0.0],
            p_mids=[self.initial_price],
            p_refs=[self.initial_price],
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
        if i1 > self.max_events:
            raise ValueError(f"Exceeded max_events={self.max_events}")

        # slice-assign each column
        self.times  [i0:i1] = times
        self.p_mids [i0:i1] = p_mids
        self.p_refs [i0:i1] = p_refs
        self.sides  [i0:i1] = sides
        self.depths [i0:i1] = depths
        self.events [i0:i1] = events
        self.redrawn[i0:i1] = redrawns

        # stack the 1D lob_states into an (n,2K) block
        self.states[i0:i1, :] = np.vstack(lob_states)

        self.step = i1

    def current_time(self):
        return self.times[self.step - 1]

    def current_mid_price(self):
        return self.p_mids[self.step - 1]

    def current_ref_price(self):
        return self.p_refs[self.step - 1]

    def current_state(self, history_size=1):
        return self.states[self.step - history_size:self.step].copy()

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
            next_t, self.inv_bid, self.inv_ask, self.max_events_intra
        )

        # JIT returns numeric codes for side/depth/event,
        # so we can write them directly
        self._write_batch(
            times, p_mids, p_refs,
            sides, depths, events,
            redrawns, lob_states
        )


    def to_dataframe(self):
        """
            Convert the logged simulation into a Pandas DataFrame.
            Only used for debugging and visualization.
        """
        import pandas as pd

        df = pd.DataFrame({
            'time':   self.times[:self.step],
            'p_mid':  self.p_mids[:self.step],
            'p_ref':  self.p_refs[:self.step],
            'side':   self.sides[:self.step],
            'depth':  self.depths[:self.step],
            'event':  self.events[:self.step],
            'redrawn':self.redrawn[:self.step]
        })

        df['side']  = df['side'].map({1:'bid', 2:'ask'})
        df['event'] = df['event'].map({1:'limit', 2:'cancel', 3:'market', 4:'trader'})
        df['redrawn'] = df['redrawn'].astype(bool)

        bids = [f"q_bid{i+1}" for i in range(self.K)][::-1]
        asks = [f"q_ask{i+1}" for i in range(self.K)]
        cols = bids + asks
        block = self.states[:self.step, :]
        for j, name in enumerate(cols):
            if j < self.K:
                df[name] = block[:, self.K - j - 1]
            else:
                df[name] = block[:, j]

        return df