import numpy as np
import torch 
import torch.nn as nn
from numba import njit
import torch.optim as optim
from collections import deque, namedtuple
import wandb
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from contextlib import nullcontext

from simulate_qrm_numba import IntensityTable, sample_stationary_lob, simulate_QRM_jit, update_LOB


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
        max_events_intra: int = 200
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
        self.sides   = np.zeros(self.max_events, np.int32) # 1 = bid, 2 = ask
        self.depths  = np.zeros(self.max_events, np.int32) # depth of the event (nb of executed shares when trader action)
        self.events  = np.zeros(self.max_events, np.int32) # 1 = limit, 2 = cancel, 3 = order, 4 = trader
        self.redrawn = np.zeros(self.max_events, np.bool_) # 0 = not redrawn, 1 = redrawn
        self.states  = np.empty((self.max_events, 2*self.K), np.int32) # [q_bid1, ..., q_bidK, q_ask1, ..., q_askK] format

        # sample from invariant distribution
        lob0 = np.empty(2*self.K, np.int32)
        lob0[:self.K]   = sample_stationary_lob(self.inv_bid, np.empty((0,), np.int32))
        lob0[self.K:] = sample_stationary_lob(self.inv_ask, np.empty((0,), np.int32))

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

    def current_state(self):
        return self.states[self.step - 1].copy()

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
            self.current_state(),
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
        max_events_intra: int
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
            max_events_intra = max_events_intra
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
    
    def current_state(self):
        return self.simulator.current_state()

    def best_quotes(self):
        """
            Reads the last LOB snapshot and returns 
            ((bid_price, size, depth, total_bid), (ask_price, size, depth, total_ask))
        """
        st = self.current_state()  # length 2K
        K  = self.simulator.K
        p_ref = self.current_ref_price()

        # best bid: first i where st[i]>0
        bid_idx = next((i for i in range(K) if st[i]>0), None)
        if bid_idx is None:
            raise ValueError("No best bid")
        size_bid = int(st[bid_idx])
        price_bid = p_ref - self.tick*(bid_idx + 0.5)
        total_bid = int(st[:K].sum())

        # best ask: first j where st[K+j]>0
        ask_idx = next((j for j in range(K) if st[K+j]>0), None)
        if ask_idx is None:
            raise ValueError("No best ask")
        size_ask = int(st[K+ask_idx])
        price_ask = p_ref + self.tick*(ask_idx + 0.5)
        total_ask = int(st[K:].sum())

        return (price_bid, size_bid, bid_idx+1, total_bid), \
               (price_ask, size_ask, ask_idx+1, total_ask)

    def get_state(self):

        (bid_price, bid_size, *_), (ask_price, ask_size, *_) = self.best_quotes()
        
        if self.simulator.next_trader_time_idx < len(self.trader_times):
            nxt = self.trader_times[self.simulator.next_trader_time_idx]
        else: # boundary case 
            nxt = self.trader_times[-1] + self.step_trader_times

        return {
            'inventory': self.current_inventory,
            'time':      nxt,
            'best_ask_price_1': ask_price,
            'best_ask_size_1':  ask_size,
            'best_bid_size_1':  bid_size
        }

    def state_to_vector(self, st: dict):
        """
            Normalization for neural network input.
        """
        inv  = st['inventory']
        t    = st['time']
        ap   = st['best_ask_price_1']
        asz  = st['best_ask_size_1']
        bsz  = st['best_bid_size_1']

        inv_n = 2*inv / self.initial_inventory - 1
        t_n   = 2*t / self.time_horizon - 1
        ap_n  = (ap - self.arrival_price - self.price_offset) / self.price_std
        asz_n = (asz - self.vol_offset) / self.vol_std
        bsz_n = (bsz - self.vol_offset) / self.vol_std

        return np.array([inv_n, t_n, ap_n, asz_n, bsz_n], dtype=np.float64)

    def step(self, action: int):
        """
            Apply trader `action` (size to take on the ask side),
            then simulate QRM to the next trader time, compute reward/break.
        """
        nxt     = self.trader_times[self.simulator.next_trader_time_idx]
        st      = self.current_state()
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
        rat = self.risk_aversion * self.current_inventory / (self.time_horizon + 2 - nxt)
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
                self.tick, self.inv_bid, self.inv_ask
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

        return self.get_state(), reward, done


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """
        Experience replay buffer for storing Transitions.
    """

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    """
        Neural network for approximating the Q-values.
    """

    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        ### --- Ning et al. NN ---
        # self.fc = nn.Sequential(
        #     nn.Linear(state_dim, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, action_dim)
        # )
        ### --- NN 1 ---- 
        # self.fc = nn.Sequential(
        #     nn.Linear(state_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, action_dim)
        # )
        ### --- NN 2 ----
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)



class DDQNAgent:
    """
        Double Deep Q-Network (DDQN) agent (see Van Hasselt et al. (2016)).
    """

    def __init__(self, state_dim=1, action_dim=1, device='cpu',
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3,
                 alpha=0.95, eps=0.01, proba_0=0.8):
        
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.proba_0 = proba_0
        
        # Policy and target networks
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=self.lr,       
            alpha=self.alpha,     
            eps=self.eps          
            )
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.exploration_mode = 'rl'
        
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        
        self.loss_fn = nn.SmoothL1Loss()

        # === Trading strategies ===
        self.fixed_action = 2
        self.twap_agent = None
        self.backload_agent = None


    def select_action(self, state, episode):
        """ 
            ε-greedy action selection.
            `state` is supposed normalized by the function `state_to_vector`.
        """
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.exploration_mode == 'rl':
            if np.random.random() < self.epsilon:
                actions = [i for i in range(self.action_dim)]
                probs = [self.proba_0] + [(1-self.proba_0) / (self.action_dim - 1)] * (self.action_dim - 1)
                return np.random.choice(actions, p=probs)
            else:
                with torch.no_grad():
                    return self.policy_net(state_tensor).argmax().item()
        
        elif self.exploration_mode == 'front_load':
            return self.fixed_action
        
        elif self.exploration_mode == 'back_load':
            self.backload_agent.fixed_action = self.fixed_action
            self.backload_agent.exec_steps = int(np.ceil(self.backload_agent.initial_inventory / self.fixed_action))
            return self.backload_agent.select_action(state, episode)
        
        elif self.exploration_mode == 'twap':
            return self.twap_agent.select_action(state, episode)
        
        else:
            raise ValueError(f"Unknown exploration mode: {self.exploration_mode}")
            

    def store_transition(self, state, action, reward, next_state, done):
        """ 
            Store transition in the replay memory.
            `state` and `next_state` is supposed normalized by the function `state_to_vector`.
        """
        self.memory.push(
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done
        )


    def learn(self):
        """
            Sample a batch of transitions from the replay memory and update the policy network.
        """

        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            max_next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)
        
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        """
            Update the target network with the weights of the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Burn-in phase: not used at the moment
def generate_boundary_episodes(env, strategy="sell_first", n_pretrain_paths=100):

    episodes = []
    
    for _ in range(n_pretrain_paths):

        state = env.reset()
        # state_vec = state_to_vector(state)
        
        (state)
        transitions = []
        done = False

        while not done:

            if strategy == "sell_first":
                action = 4 
            elif strategy == "sell_last":
                if env.current_time < env.time_horizon - 25:
                    action = 0
                else:
                    action = 4
            elif strategy == "do_nothing":
                action = 0
            
            next_state, reward, done = env.step(action)
            # next_state_vec = state_to_vector(next_state)
            # transitions.append((state_vec, action, reward, next_state_vec, done))

        episodes.extend(transitions)

    return episodes


def save_model(agent, path):
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict()
    }, path)


def load_model(agent, path, test_mode=False):
    if not test_mode:
        if type(path) == str:
            checkpoint = torch.load(path, weights_only=False)
            agent.policy_net.load_state_dict(checkpoint['policy_net'])
            agent.target_net.load_state_dict(checkpoint['target_net'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            return 
    else:
        if not path:
            raise ValueError("Must provide load_model_path in test mode.")
        checkpoint = torch.load(path, weights_only=False)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['policy_net'])  
        agent.optimizer = None
        agent.policy_net.eval()
        agent.target_net.eval()
        agent.epsilon = 0.0



exploration_mode_dic = {'rl': int(0), 'front_load': int(1), 'back_load': int(2), 'twap': int(3)}

def train(episodes, risk_aversion=1, proba_0=0.2):

    ### use: train(episodes=10000, risk_aversion=0, proba_0=1-0.025)
    
    # === Simulation Parameters ===
    time_horizon = 300
    initial_inventory = 75
    actions = [0, 1, 2, 3, 4]
    proba_0 = proba_0
    max_events = 20000
    max_events_intra = 200

    price_offset = 0.
    price_std = 0.1
    vol_offset = 5      # 4 for asym
    vol_std = 4         # 3.5 for sym

    theta = 0.6
    theta_reinit = 0.85
    tick = 0.01
    arrival_price = 100.005
    trader_times = np.arange(0, int(time_horizon)+1, 1.0)
    final_penalty = 1.0
    risk_aversion = risk_aversion

    # === Others ===
    state_dim = 5
    action_dim = len(actions)
    logging_every = 5
    rl_agent_type = 'ddqn'

    # === Epsilons ===
    epsilon_start = 1.0 
    epsilon_decay = 0.995
    epsilon_end = 0.01

    # === NN hyper parameters ===
    gamma = 0.99
    learning_rate = 0.001
    alpha = 0.95
    eps = 0.01
    batch_size = 64
    memory_capacity = 10000 # as in Ning et al.
    target_update_freq = 3000 # Ning et al: 15
    burn_in = False

    # === Invariant Distribution ===
    file_name_bid = 'aapl_corrected.npy' 
    file_name_ask = 'aapl_corrected.npy' 
    file_name_bid = 'aapl_price_down_bid.npy' 
    file_name_ask = 'aapl_price_down_ask.npy' 
    folder_path_invariant_distrib = 'calibration_data/invariant_distribution/'
    inv_bid_file = folder_path_invariant_distrib + file_name_bid
    inv_ask_file = folder_path_invariant_distrib + file_name_ask

    # === Intensity Table ===
    file_name = 'aapl_corrected.npy'
    file_name = 'aapl_price_down.npy'
    folder_path_intensity_table = 'calibration_data/intensity_table/' 
    intensity_table_array = np.load(folder_path_intensity_table + file_name)
    K, Q_plus_1, *_ = intensity_table_array.shape
    Q = Q_plus_1 - 1
    inten_table = IntensityTable(max_depth=K, max_queue=Q)
    inten_table._data = intensity_table_array

    # === Load pre-trained model ===
    load_model_bool = False
    load_model_path = '.pth'

    ######### END OF PARAMETERS #########


    # === Initialize Weights & Biases ===
    wandb.init(
        project="QRM_RL_Agent",
        name=f"{rl_agent_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            # Simulation Parameters
            "rl_agent_type": rl_agent_type,
            "episodes": episodes,
            "time_horizon": time_horizon,
            "initial_inventory": initial_inventory,
            "price_offset": price_offset,
            "price_std": price_std,
            "vol_offset": vol_offset,
            "vol_std": vol_std,
            "theta": theta,
            "theta_reinit": theta_reinit,
            "final_penalty": final_penalty,
            "risk_aversion": risk_aversion,

            # Epsilon Parameters
            "proba_0": proba_0,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,

            # Neural Network Hyperparameters
            "gamma": gamma,
            "learning_rate": learning_rate,
            "alpha": alpha,
            "eps": eps,
            "batch_size": batch_size,
            "memory_capacity": memory_capacity,
            "target_update_freq": target_update_freq,
            "burn_in": burn_in,

            # Intensity Table
            "file_name": file_name,

            # Model Loading
            "load_model_bool": load_model_bool,
            "load_model_path": load_model_path
        }
    )
    run_id = wandb.run.id
    wandb.run.name = f"{rl_agent_type}_{run_id}"

    # === Environment Initialization ===
    env = MarketEnvironment(
        intensity_table=inten_table,
        actions=actions,
        theta=theta,
        theta_reinit=theta_reinit,
        tick=tick,
        arrival_price=arrival_price,
        inv_bid_file=inv_bid_file,
        inv_ask_file=inv_ask_file,
        trader_times=trader_times,
        initial_inventory=initial_inventory,
        time_horizon=time_horizon,
        final_penalty=final_penalty,
        risk_aversion=risk_aversion, 
        price_offset=price_offset,
        price_std=price_std,
        vol_offset=vol_offset,
        vol_std=vol_std, 
        max_events=max_events, 
        max_events_intra=max_events_intra
    )

    # === Agent Initialization ===
    agent = DDQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_capacity=memory_capacity,
        batch_size=batch_size,
        gamma=gamma,    
        lr=learning_rate,
        alpha=alpha, 
        eps=eps, 
        proba_0=proba_0
    )

    twap_agent = TWAPAgent(
            time_horizon=time_horizon,
            initial_inventory=initial_inventory,
            trader_time_step=trader_times[1] - trader_times[0]
        )
    agent.twap_agent = twap_agent
    backload_agent = BackLoadAgent(
        time_horizon=time_horizon,
        initial_inventory=initial_inventory,
        trader_time_step=trader_times[1] - trader_times[0], 
        fixed_action=1 # this will be modified accordingly during training
    ) 
    agent.backload_agent = backload_agent

    # === Load pre-trained weights if specified ===
    if load_model_bool:
        load_model(agent, load_model_path)
        print(f"Model loaded from {load_model_path}")

    # === Burn-in phase ===
    if burn_in:
        pass

    # === Book-keeping ===
    episodes_rewards = []
    detailed_rewards = {}
    final_inventories = []
    episode_lengths = []
    final_is = []
    l_returns = []
    p_mids = {}

    # Python seed (RL agent)
    np.random.seed(42) 
    # Numba seed (QRM simulator)
    @njit
    def initialize_numba_rng(seed):
        np.random.seed(seed)
    initialize_numba_rng(42)

    count = 0
    for ep in range(episodes):

        # === Environment initialization ===
        state = env.reset()
        state_vec = env.state_to_vector(state)

        # === Epsilon policy ===
        if ep < int(0.2* episodes):
            agent.exploration_mode = np.random.choice(
                ['front_load', 'twap', 'back_load'], p=[0.3, 0.4, 0.3])
            agent.epsilon = 0.0
            fixed_action = np.random.choice(
                [i for i in range(1, agent.action_dim)]
            )
            agent.fixed_action = fixed_action
        else:
            agent.exploration_mode = 'rl'
            if ep < int(0.8* episodes):
                # linear decay
                a = (epsilon_end - epsilon_start) / (0.6 * episodes)
                b = epsilon_start - 0.2 * episodes * a
                agent.epsilon = a * ep + b
            else:
                agent.epsilon = epsilon_end

        # === Logging ===
        done, ep_reward, rewards = False, 0.0, []
        actions_taken = []

        while not done:

            # === Action and simulation step ===
            action = agent.select_action(state_vec, ep) 
            next_state, reward, done = env.step(action)
            next_state_vec = env.state_to_vector(next_state)

            # === Learn ===
            agent.store_transition(state_vec, action, reward, next_state_vec, done)
            agent.learn()
            # === Update target network ===
            if count % target_update_freq == 0 and count > 0:
                agent.update_target_network()
            count += 1

            # === Logging ===
            state_vec = next_state_vec
            ep_reward += reward
            actions_taken.append(action)
            # rewards.append(reward)

            wandb.log({
                "Inventory Normalized": state_vec[0],
                "Time Normalized": state_vec[1],
                "Best Ask Price Normalized": state_vec[2],
                "Best Ask Size Normalized": state_vec[3],
                "Best Ask Size": next_state['best_ask_size_1'],
                "Best Ask Price 1": next_state['best_ask_price_1'],
                "Reward": reward, 
                "Risk Aversion Term in Reward": env.risk_aversion_term,
                "Inventory": env.current_inventory, 
                "Implementation Shortfall": env.current_is,
                "Action": action,
                "Mid Price": env.current_mid_price()
            })

        # === Print progress ===
        if ep % logging_every == 0:
            print(f"Episode {ep+1}/{episodes}, Reward: {ep_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

        # === End-of-episode book-keeping ===
        # detailed_rewards[ep] = rewards
        # final_inventories.append(env.current_inventory)
        # episode_lengths.append(env.current_time())
        # episodes_rewards.append(ep_reward)
        # final_is.append(env.final_is)
        # p_mids[ep] = env.simulator.p_mids[:env.simulator.step].copy()
        # LOB = env.simulator.current_LOB
        # LOB_trades = LOB[LOB.time.isin(env.trader_times[1:])]
        # returns = LOB_trades.p_mid.pct_change().fillna(0).values
        # l_returns.append(returns)

        # === Wandb Logging ===
        wandb.log({
            "Episode": ep,
            "Final Reward": ep_reward,
            "Epsilon": agent.epsilon,
            "Final Inventory": env.current_inventory,
            "Final Implementation Shortfall": env.final_is,
            "Episode Length": env.current_time(),
            **{f"Action_{a}_count": actions_taken.count(a) for a in range(agent.action_dim)}, 
            "Non Executed Liquidity Constraint": env.non_executed_liquidity_constraint, 
            "Exploration Mode": exploration_mode_dic[agent.exploration_mode],
            "Fixed Action": agent.fixed_action
        })


    # === Save weights of NNs and optimizer ===
    print("\n[INFO] Saving model...")
    save_model(agent, path=f'save_model/{rl_agent_type}_{run_id}.pth')
    print(f"[INFO] Model saved at episode {ep}. Exiting cleanly.")

    # === Finish wandb run ===
    # all_returns = np.concatenate(l_returns).flatten()
    # table = wandb.Table(data=[[i, v] for i, v in enumerate(all_returns)], columns=["index", "return"])
    # wandb.log({"Returns between Trader Times": table})
    wandb.finish()


class InactiveAgent:
    """
        Inactive agent that does not take any action.
    """
    def select_action(self, state, episode):
        return 0

class RandomAgent:
    """
        Random agent samples uniformly from the action space.
    """
    def __init__(self, action_dim, rng):
        self.action_dim = action_dim
        self.rng = rng

    def select_action(self, state, episode):
        return self.rng.integers(self.action_dim)

class PassiveAgent:
    """
        Passive agent: 50% no action, otherwise sample uniformly from the action space.
    """
    def __init__(self, action_dim, rng):
        self.action_dim = action_dim
        self.rng = rng

    def select_action(self, state, episode):
        probs = [0.5] + [0.5 / (self.action_dim - 1)] * (self.action_dim - 1)
        return self.rng.choice([i for i in range(self.action_dim)], p=probs)
    
class AggressiveAgent:
    """
        Aggressive agent: always buys two shares.
    """
    def select_action(self, state, episode):
        return 2
    
class TWAPAgent:
    """
        Time-weighted average price (TWAP) agent.
    """

    def __init__(self, time_horizon, initial_inventory, trader_time_step):
        self.time_horizon = time_horizon
        self.initial_inventory = initial_inventory
        self.trader_time_step = trader_time_step
        ratio = int(initial_inventory / time_horizon / trader_time_step)
        self.actions = self.distribute_ones(int(time_horizon/trader_time_step), initial_inventory, ratio)

    @staticmethod
    def distribute_ones(n, n_0, ratio):
        """
            Uniformly distribute n_0 ones in an array of size n.
        """
        arr = ratio * np.ones(n, dtype=int)
        new_n_0 = n_0 % n
        for i in range(new_n_0):
            index = round(i * (n - 1) / (new_n_0 - 1)) if new_n_0 > 1 else n // 2
            arr[index] += 1
        assert np.sum(arr) == n_0, "TWAP Error: The number of ones in the array does not match n_0."
        return arr

    def select_action(self, state, episode):
        time_norm = state[1]
        time = (time_norm + 1)*self.time_horizon / 2
        idx = round(time / self.trader_time_step) - 1 
        assert idx < len(self.actions), "TWAP Error: Index out of bounds."      
        return self.actions[idx] 


class BackLoadAgent:
    """
        Back-loaded execution agent with a fixed per-step execution size.
        Waits until the last possible slots, then executes `fixed_action` each
        decision, ensuring full liquidation by the final step.
    """

    def __init__(
        self,
        time_horizon: float,
        initial_inventory: int,
        trader_time_step: float, 
        fixed_action: int = 1, 
        security_margin: int = 0
    ):
        self.time_horizon      = time_horizon
        self.initial_inventory = initial_inventory
        self.trader_time_step  = trader_time_step
        self.fixed_action      = fixed_action
        self.security_margin   = security_margin # start slightly earlier to ensure full liquidation because of QRM liquidity constraints
        self.n_steps = int(np.ceil(time_horizon / trader_time_step))
        self.exec_steps = int(np.ceil(initial_inventory / fixed_action))


    def select_action(self, state, episode):
        
        time_norm = state[1]
        curr_time = (time_norm + 1) * self.time_horizon / 2 # inverse transform of state normalization
        idx = int(np.floor(curr_time / self.trader_time_step))

        if idx >= self.n_steps - self.exec_steps - self.security_margin: 
            return self.fixed_action
        else:
            return 0


class FrontLoadAgent:
    def __init__(self, fixed_action):
        self.fixed_action = fixed_action
    
    def select_action(self, state, episode):
        return self.fixed_action
        



def test(episodes, risk_aversion, agent, load_model_path=None):
    
    # === Simulation Parameters ===
    time_horizon = 300
    initial_inventory = 75
    actions = [0, 1, 2, 3, 4]
    proba_0 = 1 - 0.025
    max_events = 20000
    max_events_intra = 200

    price_offset = 0.
    price_std = 0.1
    vol_offset = 5      # 5 for sym
    vol_std = 4       # 4 for sym

    theta = 0.6
    theta_reinit = 0.85
    tick = 0.01
    arrival_price = 100.005
    trader_times = np.arange(0, int(time_horizon)+1, 1.0)
    final_penalty = 1
    risk_aversion = risk_aversion

    # === Others ===
    state_dim = 5
    action_dim = len(actions)
    logging_every = 5
    agent_name_map = {
        DDQNAgent: 'ddqn', 
        InactiveAgent: 'inactive', 
        RandomAgent: 'random',
        PassiveAgent: 'passive',
        TWAPAgent: 'twap', 
        BackLoadAgent: 'back_load',
        FrontLoadAgent: 'front_load'
    }
    agent_type = agent_name_map.get(type(agent), 'Unknown')

    # === Epsilons ===
    epsilon_start = 1.0 
    epsilon_decay = 0.995
    epsilon_end = 0.01

    # === NN hyper parameters ===
    gamma = 0.99
    learning_rate = 0.001
    alpha = 0.95
    eps = 0.01
    batch_size = 64
    memory_capacity = 10000
    target_update_freq = 15 
    burn_in = False

    # === Invariant Distribution ===
    file_name_bid = 'aapl_corrected.npy' 
    file_name_ask = 'aapl_corrected.npy' 
    folder_path_invariant_distrib = 'calibration_data/invariant_distribution/'
    inv_bid_file = folder_path_invariant_distrib + file_name_bid
    inv_ask_file = folder_path_invariant_distrib + file_name_ask

    # === Intensity Table ===
    file_name = 'aapl_corrected.npy'
    folder_path_intensity_table = 'calibration_data/intensity_table/' 
    intensity_table_array = np.load(folder_path_intensity_table + file_name)
    K, Q_plus_1, *_ = intensity_table_array.shape
    Q = Q_plus_1 - 1
    inten_table = IntensityTable(max_depth=K, max_queue=Q)
    inten_table._data = intensity_table_array

    ######### END OF PARAMETERS #########


    # === Initialize Weights & Biases ===
    wandb.init(
        project="QRM_RL_Agent",
        name=f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config={

            # Simulation Parameters
            "agent_type": agent_type,
            "episodes": episodes,
            "time_horizon": time_horizon,
            "initial_inventory": initial_inventory,
            "price_offset": price_offset,
            "price_std": price_std,
            "vol_offset": vol_offset,
            "vol_std": vol_std,
            "theta": theta,
            "theta_reinit": theta_reinit,
            "final_penalty": final_penalty,
            "risk_aversion": risk_aversion,

            # Epsilon Parameters
            "proba_0": proba_0,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,

            # Neural Network Hyperparameters
            "gamma": gamma,
            "learning_rate": learning_rate,
            "alpha": alpha,
            "eps": eps,
            "batch_size": batch_size,
            "memory_capacity": memory_capacity,
            "target_update_freq": target_update_freq,
            "burn_in": burn_in,

            # Intensity Table
            "file_name": file_name
        }
    )
    run_id = wandb.run.id
    wandb.run.name = f"{agent_type}_{run_id}"

    # === Environment and Agent Initialization ===
    env = MarketEnvironment(
        intensity_table=inten_table,
        actions=actions,
        theta=theta,
        theta_reinit=theta_reinit,
        tick=tick,
        arrival_price=arrival_price,
        inv_bid_file=inv_bid_file,
        inv_ask_file=inv_ask_file,
        trader_times=trader_times,
        initial_inventory=initial_inventory,
        time_horizon=time_horizon,
        final_penalty=final_penalty,
        risk_aversion=risk_aversion,
        price_offset=price_offset,
        price_std=price_std,
        vol_offset=vol_offset,
        vol_std=vol_std,
        max_events=max_events, 
        max_events_intra=max_events_intra
    )

    # === Agent Initialization ===
    if isinstance(agent, DDQNAgent):
        agent = DDQNAgent(
            state_dim=state_dim, 
            action_dim=action_dim,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            memory_capacity=memory_capacity,
            batch_size=batch_size,
            gamma=gamma,    
            lr=learning_rate,
            alpha=alpha, 
            eps=eps, 
            proba_0=proba_0
        )
        # === Load pre-trained weights if specified ===
        if load_model_path is None:
            raise ValueError("Please provide a path to load the model.")
        else:
            load_model(agent, load_model_path, test_mode=True)
            print(f"Model loaded from {load_model_path}")
        # === Test mode ===
        agent.policy_net.eval()
        agent.target_net.eval()
        agent.epsilon = 0.0

    # === Book-keeping ===
    episodes_rewards = []
    detailed_rewards = {}
    final_inventories = []
    episode_lengths = []
    final_is = []
    mid_prices = {}
    actions = {}
    idx_actions = {}
    LOBs = {}
    l_returns = []

    # Python seed (RL agent)
    np.random.seed(2025) 
    # Numba seed (QRM simulator)
    @njit
    def initialize_numba_rng(seed):
        np.random.seed(seed)
    initialize_numba_rng(2025)

    for ep in range(episodes):

        # Environment initialization
        state = env.reset()
        state_vec = env.state_to_vector(state)

        done, ep_reward, rewards = False, 0.0, []
        actions_taken = []

        with torch.no_grad():
            while not done:

                # === Action and simulation step ===
                action = agent.select_action(state_vec, ep) 
                next_state, reward, done = env.step(action)
                next_state_vec = env.state_to_vector(next_state)

                # === Logging ===
                state_vec = next_state_vec
                ep_reward += reward
                actions_taken.append(action)
                # rewards.append(reward)

                wandb.log({
                    "Inventory Normalized": state_vec[0],
                    "Time Normalized": state_vec[1],
                    "Best Ask Price Normalized": state_vec[2],
                    "Best Ask Size Normalized": state_vec[3],
                    "Best Ask Size": next_state['best_ask_size_1'],
                    "Best Ask Price 1": next_state['best_ask_price_1'],
                    "Reward": reward, 
                    "Risk Aversion Term in Reward": env.risk_aversion_term,
                    "Inventory": env.current_inventory, 
                    "Implementation Shortfall": env.current_is,
                    "Action": action,
                    "Mid Price": env.current_mid_price()
                })

        # === Print progress ===
        if ep % logging_every == 0:
            print(f"Episode {ep+1}/{episodes}")

        # === End-of-episode book-keeping ===
        final_is.append(env.final_is)
        mid_prices[ep] = env.simulator.p_mids[:env.simulator.step]
        actions[ep] = actions_taken
        all_times = env.simulator.times[:env.simulator.step]
        idx_actions[ep] = np.where(np.isin(all_times, env.trader_times[1:]))[0].tolist()
        # detailed_rewards[ep] = rewards
        # final_inventories.append(env.current_inventory)
        # episode_lengths.append(env.current_time())
        # episodes_rewards.append(ep_reward)
        # returns = LOB_trades.p_mid.pct_change().fillna(0).values
        # l_returns.append(returns)
        # LOBs[ep] = LOB

        # === Wandb Logging ===
        wandb.log({
            "Episode": ep,
            "Final Reward": ep_reward,
            "Final Inventory": env.current_inventory,
            "Final Implementation Shortfall": env.final_is,
            "Episode Length": env.current_time(),
            **{f"Action_{a}_count": actions_taken.count(a) for a in range(action_dim)}, 
            "Non Executed Liquidity Constraint": env.non_executed_liquidity_constraint
        })

    # === Finish wandb run ===
    # all_returns = np.concatenate(l_returns).flatten()
    # table = wandb.Table(data=[[i, v] for i, v in enumerate(all_returns)], columns=["index", "return"])
    # wandb.log({"Returns between Trader Times": table})
    wandb.finish()

    dic = {'final_is': final_is, 
           'mid_prices': mid_prices,
           'actions': actions,
           'idx_actions': idx_actions
           }
    
    return dic, run_id











class RLRunner:
    def __init__(self, config, load_model_path=None):
        # Unpack config
        self.cfg = config
        self.mode = config['mode']
        self.episodes = config['episodes']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = None
        self.agent_name_map = {
            DDQNAgent: 'ddqn', 
            InactiveAgent: 'inactive', 
            RandomAgent: 'random',
            PassiveAgent: 'passive',
            TWAPAgent: 'twap', 
            BackLoadAgent: 'back_load',
            FrontLoadAgent: 'front_load'
        }

        # Seeds
        np.random.seed(config['seed'])
        @njit
        def _init_numba(seed): np.random.seed(seed)
        _init_numba(config['seed'])

        # WandB init
        wandb.init(
            project="QRM_RL_Agent",
            name=f"{self.mode}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            config=config
        )

        # Build intensity table
        inten_arr = np.load(config['folder_path_intensity_table'] + config['file_name'])
        K, Qp1, *_ = inten_arr.shape
        inten_table = IntensityTable(max_depth=K, max_queue=Qp1-1)
        inten_table._data = inten_arr

        # Environment
        self.env = MarketEnvironment(
            intensity_table=inten_table,
            actions=config['actions'],
            theta=config['theta'],
            theta_reinit=config['theta_reinit'],
            tick=config['tick'],
            arrival_price=config['arrival_price'],
            inv_bid_file=config['folder_path_invariant'] + config['file_name_bid'],
            inv_ask_file=config['folder_path_invariant'] + config['file_name_ask'],
            trader_times=config['trader_times'],
            initial_inventory=config['initial_inventory'],
            time_horizon=config['time_horizon'],
            final_penalty=config['final_penalty'],
            risk_aversion=config['risk_aversion'],
            price_offset=config['price_offset'],
            price_std=config['price_std'],
            vol_offset=config['vol_offset'],
            vol_std=config['vol_std'],
            max_events=config['max_events'],
            max_events_intra=config['max_events_intra']
        )

        # Agent
        self.agent = DDQNAgent(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            device=self.device,
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay'],
            memory_capacity=config['memory_capacity'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            lr=config['learning_rate'],
            alpha=config['alpha'],
            eps=config['eps'],
            proba_0=config['proba_0']
        )

        if self.mode == 'train':
            self.agent.twap_agent = TWAPAgent(
                time_horizon=config['time_horizon'],
                initial_inventory=config['initial_inventory'],
                trader_time_step=config['trader_time_step']
            )
            self.agent.backload_agent = BackLoadAgent(
                time_horizon=config['time_horizon'],
                initial_inventory=config['initial_inventory'],
                trader_time_step=config['trader_time_step'], 
                fixed_action=1
            )

        # Load weights
        test_mode = (self.mode == 'test')
        load_model(self.agent, load_model_path, test_mode=test_mode)

    def _update_epsilon(self, ep):
        E = self.episodes
        bs = int(0.2 * E)
        ge = int(0.8 * E)
        if ep < bs:
            self.agent.exploration_mode = np.random.choice(
                ['front_load','twap','back_load'], p=[0.3,0.4,0.3]
            )
            self.agent.epsilon = 0.0
            self.agent.fixed_action = np.random.choice(
                [a for a in range(1, self.agent.action_dim)]
            )
        elif ep < ge:
            self.agent.exploration_mode = 'rl'
            decay_steps = ge - bs
            eps_start, eps_end = self.cfg['epsilon_start'], self.cfg['epsilon_end']
            a = (eps_end - eps_start) / decay_steps
            b = eps_start - bs * a
            self.agent.epsilon = a * ep + b
        else:
            self.agent.exploration_mode = 'rl'
            self.agent.epsilon = self.cfg['epsilon_end']

    def _log_step(self, state_vec, next_state, reward, action):
        d = {
            "Inventory Normalized": state_vec[0],
            "Time Normalized": state_vec[1],
            "Best Ask Price Normalized": state_vec[2],
            "Best Ask Size Normalized": state_vec[3],
            "Best Ask Size": next_state['best_ask_size_1'],
            "Best Ask Price 1": next_state['best_ask_price_1'],
            "Reward": reward, 
            "Risk Aversion Term in Reward": self.env.risk_aversion_term,
            "Inventory": self.env.current_inventory, 
            "Implementation Shortfall": self.env.current_is,
            "Action": action,
            "Mid Price": self.env.current_mid_price()
        }
        wandb.log(d)

    def run(self):
        agent_type = self.agent_name_map.get(type(self.agent), 'Unknown')
        self.run_id = wandb.run.id
        wandb.run.name = f"{agent_type}_{self.run_id}"

        train_mode = (self.mode == 'train')
        if not train_mode:
            final_is = []
            mid_prices = {}
            actions_taken = {}
            idx_actions = {}
        step_count = 0
        for ep in range(self.episodes):

            state = self.env.reset()
            state_vec = self.env.state_to_vector(state)
            if train_mode:
                self._update_epsilon(ep)

            ctx = torch.no_grad() if not train_mode else nullcontext()
            with ctx:
                done, ep_reward, actions = False, 0.0, []

                while not done:
                    action = self.agent.select_action(state_vec, ep)
                    nxt, reward, done = self.env.step(action)
                    nxt_vec = self.env.state_to_vector(nxt)

                    if train_mode:
                        self.agent.store_transition(state_vec, action, reward, nxt_vec, done)
                        self.agent.learn()
                        if step_count % self.cfg['target_update_freq'] == 0:
                            self.agent.update_target_network()

                    state_vec = nxt_vec
                    ep_reward += reward
                    actions.append(action)
                    self._log_step(state_vec, nxt, reward, action)
                    step_count += 1

            # end of episode logging
            summary = {
                "Episode": ep,
                "Final Reward": ep_reward,
                "Final Inventory": self.env.current_inventory,
                "Final Implementation Shortfall": self.env.final_is,
                "Episode Length": self.env.current_time(),
                **{f"Action_{a}_count": actions.count(a) for a in range(self.cfg['action_dim'])}, 
                "Non Executed Liquidity Constraint": self.env.non_executed_liquidity_constraint, 
            }
            if train_mode:
                summary['Epsilon'] = self.agent.epsilon
                summary['Exploration Mode'] = exploration_mode_dic[self.agent.exploration_mode]
                summary['Fixed Action'] = self.agent.fixed_action
            
            wandb.log(summary)

            # === Print progress ===
            if ep % self.cfg['logging_every'] == 0:
                print(f"[{self.mode.upper()}][{ep}/{self.episodes}]  Reward={ep_reward:.2f}") #  Eps={self.agent.epsilon:.3f}")

            # === End-of-episode book-keeping ===
            if not train_mode:
                final_is.append(self.env.final_is)
                mid_prices[ep] = self.env.simulator.p_mids[:self.env.simulator.step]
                actions_taken[ep] = actions
                all_times = self.env.simulator.times[:self.env.simulator.step]
                idx_actions[ep] = np.where(np.isin(all_times, self.env.trader_times[1:]))[0].tolist()

        # save model if training
        if train_mode:
            save_model(self.agent, f"save_model/{self.agent_name_map[type(self.agent)]}_{self.run_id}.pth")

        wandb.finish()

        if not train_mode:
            dic = {
                'final_is': final_is, 
                'mid_prices': mid_prices,
                'actions': actions_taken,
                'idx_actions': idx_actions
            }
            return dic, self.run_id


# === Example usage ===

config = {
    # mode + seeding
    'mode': 'train',            # or 'test'
    'seed': 42,
    'episodes': 10000,
    'logging_every': 5,

    # simulation parameters
    'time_horizon': 300,
    'trader_time_step': 1.0,  # in seconds
    'initial_inventory': 375,
    'actions': [0,1,2,3,4],
    'max_events': 20000,
    'max_events_intra': 200,

    'price_offset': 0.,
    'price_std': 0.1,
    'vol_offset': 5,
    'vol_std': 4,
    'theta': 0.6,
    'theta_reinit': 0.85,
    'tick': 0.01,
    'arrival_price': 100.005,
    'final_penalty': 1.0,
    'risk_aversion': 0.05,

    # DQN hyperparams
    'proba_0': 0.975,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'gamma': 0.99,
    'learning_rate': 1e-3,
    'alpha': 0.95,
    'eps': 0.01,
    'batch_size': 64,
    'memory_capacity': 10000,
    'target_update_freq': 15,

    # environment/IO
    'state_dim': 5,
    'folder_path_intensity_table': 'calibration_data/intensity_table/',
    'folder_path_invariant': 'calibration_data/invariant_distribution/'
}

th = config['time_horizon']
st = config['trader_time_step']
config['trader_times'] = np.arange(0, th + st, st)
config['action_dim'] = len(config['actions'])

normal_prices = False
decreasing_prices = True
if normal_prices:
    config['file_name'] = 'aapl_corrected.npy'
    config['file_name_bid'] = 'aapl_corrected.npy'
    config['file_name_ask'] = 'aapl_corrected.npy'
if decreasing_prices:
    config['file_name'] = 'aapl_price_down.npy'
    config['file_name_bid'] = 'aapl_price_down_bid.npy'
    config['file_name_ask'] = 'aapl_price_down_ask.npy'






if __name__ == "__main__":

    ### ==== TRAIN ===== ###
    # config['episodes'] = 10000
    # runner = RLRunner(config)
    # runner.run() 



    ### ==== TEST ===== ###
    config['mode'] = 'test'
    config['seed'] = 2025
    config['episodes'] = 10
    train_run_id = 'cszrsnrn'
    runner = RLRunner(config, load_model_path=f'save_model/ddqn_{train_run_id}.pth')

    th = runner.cfg['time_horizon']
    ii = runner.cfg['initial_inventory']
    tts = runner.cfg['trader_time_step']
    max_action = max(runner.cfg['actions'])
    run_ids = [] 
    order = ['ddqn', 'twap', 'back_load', 'front_load']
    final_is = {}
    
    ### === TWAP Agent Testing === ###
    agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
    actions = agent.actions
    actions[-1] += 1 # avoid remaining inventory because of the QRM liquidity constraints
    agent.actions = actions

    runner.agent = agent
    dic, run_id = runner.run()
    run_ids.append(run_id)
    final_is['twap'] = dic['final_is']



    # ### === DDQN Agent Testing === ###
    # dic, run_id = runner.run()
    # run_ids.append(run_id)
    # final_is['ddqn'] = dic['final_is']
    # with open(f'data_wandb/dictionaries/ddqn_{run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === TWAP Agent Testing === ###
    # agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
    # actions = agent.actions
    # actions[-1] += 1 # avoid remaining inventory because of the QRM liquidity constraints
    # agent.actions = actions

    # runner.agent = agent
    # dic, run_id = runner.run()
    # run_ids.append(run_id)
    # final_is['twap'] = dic['final_is']
    # with open(f'data_wandb/dictionaries/twap_{run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Back Load Agent Testing === ###
    # agent = BackLoadAgent(time_horizon=th, initial_inventory=ii, 
    #                       trader_time_step=tts, fixed_action=max_action, security_margin=2)
    
    # runner.agent = agent
    # dic, run_id = runner.run()
    # run_ids.append(run_id)
    # final_is['back_load'] = dic['final_is']
    # with open(f'data_wandb/dictionaries/back_load_{run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Front Load Agent Testing === ###
    # agent = FrontLoadAgent(fixed_action=max_action)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # run_ids.append(run_id)
    # final_is['front_load'] = dic['final_is']
    # with open(f'data_wandb/dictionaries/front_load_{run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)


    # # === Plot Implementation Shortfall ===
    # for i, strat in enumerate(order):
    #     print(f"Run ID for {strat}: {run_ids[i]}")
    # print(f'Plot has been saved with the DDQN run ID: {run_ids[0]}')

    # plt.figure(figsize=(7, 5))
    # maxi = max(
    #     max(np.abs(np.max(values)), np.abs(np.min(values)))
    #     for values in final_is.values()
    # )
    # x = np.linspace(-maxi, maxi, 1000)
    # for key, values in final_is.items():
    #     kde = gaussian_kde(values)
    #     y_kde = kde(x)
    #     plt.plot(x, y_kde, label=key)

    # plt.xlabel('Implementation Shortfall')
    # plt.ylabel('Density')
    # plt.title('')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'plots/implementation_shortfall/{run_ids[0]}.pdf', bbox_inches='tight')
    # plt.show()
    # plt.close()