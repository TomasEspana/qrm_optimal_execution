import numpy as np
from .qrm_simulator import QueueReactiveMarketSimulator
from qrm_core.intensity import IntensityTable
from qrm_core.sampling import update_LOB


class MarketEnvironment:
    """
        RL environment for optimal execution with market interaction,
        backed by QueueReactiveMarketSimulator.
    """

    def __init__(
        self,
        intensity_table: IntensityTable,
        actions: list,
        theta: float,
        theta_reinit: float,
        tick: float,
        arrival_price: float,
        inv_bid_file: str,
        inv_ask_file: str,
        trader_times: np.ndarray,
        initial_inventory: float,
        time_horizon: float,
        final_penalty: float,
        risk_aversion: float,
        price_offset: float,
        price_std: float,
        vol_offset: float,
        vol_std: float,
        max_events: int, 
        max_events_intra: int, 
        history_size: int, 
        alpha_ramp: float, 
        basic_state: bool, 
        aes: list, 
        test_mode: bool
    ):
        # — core parameters —
        self.actions = actions
        self.arrival_price = arrival_price
        self.price_offset  = price_offset
        self.price_std     = price_std
        self.vol_offset    = vol_offset
        self.vol_std       = vol_std

        self.initial_inventory = initial_inventory
        self.current_inventory = initial_inventory
        self.time_horizon      = time_horizon
        self.final_penalty     = final_penalty
        self.risk_aversion     = risk_aversion
        self.history_size      = history_size
        self.alpha_ramp        = alpha_ramp
        self.basic_state       = basic_state
        self.aes               = aes
        self.test_mode         = test_mode

        # load intensity / inv. distributions
        self.intensity_table = np.transpose(intensity_table._data,
                                            (2,0,1,3)).copy()
        self.theta      = np.float64(theta)
        self.theta_reinit = np.float64(theta_reinit)
        self.tick       = np.float64(tick)
        self.inv_bid    = np.load(inv_bid_file)
        self.inv_ask    = np.load(inv_ask_file)
        self.trader_times = trader_times
        self.step_trader_times = self.trader_times[1] - self.trader_times[0]

        # bookkeeping for implementation shortfall
        self.current_is = 0.0
        self.final_is   = 0.0
        self.risk_aversion_term = 0.0
        self.non_executed_liquidity_constraint = 0

        # instantiate our fast, NumPy-backed simulator
        self.simulator = QueueReactiveMarketSimulator(
            intensity_table  = self.intensity_table,
            theta            = self.theta,
            theta_reinit     = self.theta_reinit,
            initial_price    = self.arrival_price,
            tick             = self.tick,
            inv_bid          = self.inv_bid,
            inv_ask          = self.inv_ask,
            trader_times     = self.trader_times,
            max_events       = max_events,
            max_events_intra = max_events_intra, 
            aes              = aes
        )

    def reset(self):
        """
            Start a fresh episode.
        """
        # reset trader
        self.current_inventory = self.initial_inventory
        self.current_is = 0.0
        self.final_is   = 0.0
        self.risk_aversion_term = 0.0
        self.non_executed_liquidity_constraint = 0

        # reset & run initial QRM up to first trader time
        self.simulator.initialize()
        self.simulator.simulate_step()

        return self.get_state()

    def current_time(self):
        return self.simulator.current_time()

    def current_mid_price(self):
        return self.simulator.current_mid_price()

    def current_ref_price(self):
        return self.simulator.current_ref_price()
    
    def current_state(self, hist_size=1):
        return self.simulator.current_state(hist_size)

    @staticmethod
    def _best_quotes(st, K, p_ref, tick):
        """
            Helper function to compute best quotes from LOB state.
        """
        # best bid: first i where st[i]>0
        bid_idx = next((i for i in range(K) if st[i]>0), None)
        if bid_idx is None:
            raise ValueError("No best bid")
        size_bid = int(st[bid_idx])
        price_bid = p_ref - tick*(bid_idx + 0.5)
        total_bid = int(st[:K].sum())
        bid_info = (price_bid, size_bid, bid_idx+1, total_bid)
    
        # best ask: first j where st[K+j]>0
        ask_idx = next((j for j in range(K) if st[K+j]>0), None)
        if ask_idx is None:
            raise ValueError("No best ask")
        size_ask = int(st[K+ask_idx])
        price_ask = p_ref + tick*(ask_idx + 0.5)
        total_ask = int(st[K:].sum())
        ask_info = (price_ask, size_ask, ask_idx+1, total_ask)

        return bid_info, ask_info


    def best_quotes(self):
        """
            Reads the last self.history_size LOB states snapshot and returns 
            ((bid_price, size, depth, total_bid), (ask_price, size, depth, total_ask))
        """
        st = self.current_state(self.history_size) # already reversed to have the latest state first
        K  = self.simulator.K
        p_ref = self.current_ref_price()

        # lob_states = np.empty((self.history_size, 2, 2), dtype=object)
        # for i in range(self.history_size):
        #     bid_info, ask_info = self._best_quotes(st[i], K, p_ref, self.tick)
        #     lob_states[i][0][:2] = ask_info[:2]  # (price_ask, size_ask)
        #     lob_states[i][1][:2] = bid_info[:2]  # (price_bid, size_bid)

        lob_states = np.empty((self.history_size, 3), dtype=object)
        for i in range(self.history_size):
            bid_info, ask_info = self._best_quotes(st[i], K, p_ref, self.tick)
            lob_states[i][0] = ask_info[0]  # (ask price)
            lob_states[i][1] = ask_info[1]  # (ask size)
            lob_states[i][2] = bid_info[1]  # (bid size)

        return lob_states.reshape(-1).tolist()

    def get_state(self):
        """ 
            Format: [
                current_inventory, 
                time,
                'best_bid_price_1',
                'best_bid_size_1':  ask_size,
                'best_ask_price_1':  bid_size, 
                'best_ask_size_1': bid_price, 
                'best_bid_price_2', etc...
            ]
        """
        lob_states = self.best_quotes()
        
        if self.simulator.next_trader_time_idx < len(self.trader_times):
            nxt = self.trader_times[self.simulator.next_trader_time_idx]
        else: # boundary case 
            nxt = self.trader_times[-1] + self.step_trader_times
        
        if self.basic_state: 
            return [self.current_inventory, nxt, lob_states[0]] #, lob_states[1], lob_states[2]] # , min(lob_states[1], self.current_inventory)] # (inv, time, ask price, ask size)
        else:
            return [self.current_inventory, nxt] + lob_states
        

    def state_to_vector(self, st):
        """
            Normalization for neural network input.
        """
        st_n = np.empty_like(st, dtype=np.float64)
        st = np.array(st)
        st_n[0] = 2 * st[0] / self.initial_inventory - 1  # inventory
        st_n[1] = 2 * st[1] / self.time_horizon - 1  # time
        # st_n[2::2] = (st[2::2] - self.arrival_price - self.price_offset) / self.price_std  # prices
        # st_n[3::2] = (st[3::2] - self.vol_offset) / self.vol_std  # volumes

        st_n[2::3] = (st[2::3] - self.arrival_price - self.price_offset) / self.price_std  # prices
        st_n[3::] = (st[3::] - self.vol_offset) / self.vol_std  # volumes
        # st_n[4::3] = (st[4::3] - self.vol_offset) / self.vol_std  # volumes

        return st_n

    @staticmethod
    def exponential_ramp(t, time_horizon, alpha, max_penalty_intra_ep=1):
        """
            Exponential ramp function for the penalty term. 
            Finishes at `max_penalty_intra_ep` (default 1) at time `time_horizon`.
        """
        numerator = np.exp(alpha * t / time_horizon) - 1
        denominator = np.exp(alpha) - 1
        return max_penalty_intra_ep * numerator / denominator

    def step(self, action: int):
        """
            Apply trader `action` (size to take on the ask side),
            then simulate QRM to the next trader time, compute reward/break.
        """
        nxt     = self.trader_times[self.simulator.next_trader_time_idx]
        st      = self.current_state()[0]
        p_ref   = self.current_ref_price()
        K       = self.simulator.K

        # enforce not empty book and inventory
        total_ask = int(st[K:].sum())
        q = min(action, total_ask-1, self.current_inventory)
        self.current_inventory -= q
        self.non_executed_liquidity_constraint += max(0, (action - total_ask + 1))

        # walk down the ask side
        rem = q
        reward = 0.0
        trade_through = False
        for depth in range(K):
            avail = int(st[K+depth])
            take  = min(rem, avail)
            if take > 0:
                if avail <= q:
                    trade_through = True
                st[K+depth] -= take
                rem         -= take
                reward      += (self.arrival_price - (p_ref + self.tick * (depth + 0.5))) * take
            if rem == 0:
                break

        # IS accounting
        self.current_is = reward
        self.final_is  += reward

        # risk aversion term
        rat = self.risk_aversion * self.current_inventory / self.initial_inventory # * self.exponential_ramp(nxt, self.time_horizon, self.alpha_ramp)
        reward -= rat
        self.risk_aversion_term = rat

        if not trade_through:
            self.simulator._write_batch(
                times=[nxt],
                p_mids=[self.current_mid_price()],
                p_refs=[p_ref],
                sides=[2],           # 2 = ask
                depths=[q],
                events=[4],          # 4 = trader
                redrawns=[False],
                lob_states=[st]
            )
        else:
            p_mid, p_ref, st, redrawn = update_LOB(
                K, p_ref, st, 1, self.theta, self.theta_reinit,
                self.tick, self.inv_bid, self.inv_ask, self.aes
            )
            self.simulator._write_batch(
                times=[nxt],
                p_mids=[p_mid],
                p_refs=[p_ref],
                sides=[2],           # 2 = ask
                depths=[q],
                events=[4],          # 4 = trader
                redrawns=[redrawn],
                lob_states=[st]
            ) 

        # check termination
        done = False
        if nxt < self.time_horizon and self.current_inventory > 0:
            self.simulator.simulate_step()  
        else:
            done = True
            reward -= self.final_penalty * self.current_inventory
            self.simulator.next_trader_time_idx += 1

            if self.test_mode: # execute remaining inventory aggressively (test mode only)
                q_bis = self.current_inventory
                if q_bis > 0:
                    st      = self.current_state()[0]
                    p_ref   = self.current_ref_price()
                    # walk down the ask side
                    rem = q_bis
                    rwd = 0.0
                    for depth in range(K):
                        avail = int(st[K+depth])
                        take  = min(rem, avail)
                        if take > 0:
                            rem -= take
                            rwd += (self.arrival_price - (p_ref + self.tick * (depth + 0.5))) * take
                        if rem == 0:
                            break
                    if rem > 0:
                        rwd += (self.arrival_price - (p_ref + self.tick * (depth + 1.5))) * rem

                    self.final_is += rwd

        return self.get_state(), reward, done, q, total_ask