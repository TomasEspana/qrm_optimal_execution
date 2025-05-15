import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import wandb
from datetime import datetime

from simulate_qrm import IntensityTable, sample_stationary_lob_from_file, simulate_QRM, update_LOB




class QueueReactiveMarketSimulator:
    """
        Class for simulating the queue-reactive market (QRM) model.
        See Section 3.1 of Huang et al. (2015) for details.
        This implements Model III (Section 3.1) where queues are supposed independent.
    """
    def __init__(
        self,
        intensity_table: IntensityTable,
        theta: float,
        theta_reinit: float, 
        initial_price: float,
        rng: np.random.Generator,
        tick: float,
        inv_distrib_file: str,
        trader_times: np.ndarray,
        bid_ask_sym: bool
    ):
        """
            Initializes the QRM simulator.
        """
        self.intensity_table = intensity_table                 # table of estimated intensities
        K, Q_plus_1, *_ = self.intensity_table._data.shape
        self.K = K                                             # maximum depth (same as in Huang et al. (2015))
        self.Q = Q_plus_1 - 1                                  # maximum queue size
        self.theta = theta                                     # same as in Huang et al. (2015)
        self.theta_reinit = theta_reinit                       # same as in Huang et al. (2015)
        self.tick = tick                                       # tick size
        self.trader_times = trader_times                       # trader decision times (in seconds)
        self.inv_distrib_file = inv_distrib_file               # file to sample invariant distribution
        self.initial_price = initial_price                     # initial price of the simulation
        self.rng = rng                                         # random number generator for reproducibility
        self.bid_ask_sym = bid_ask_sym                         # boolean for bid-ask symmetry

    def initialize(self):
        """
            Draws the initial LOB state from the invariant distribution (Section 2.3.3).
            We do not necessarily suppose bid-ask symmetry of the intensities.
            This leads to a case distinction in the code and potentially drawing from
            two different invariant distributions.
        """
        lob_state = np.zeros(2*self.K, dtype=int) # [q_bid1, ..., q_bidK, q_ask1, ..., q_askK] format 
        if not self.bid_ask_sym:
            lob_state[:self.K] = sample_stationary_lob_from_file('bid', self.inv_distrib_file, self.rng)
            lob_state[self.K:] = sample_stationary_lob_from_file('ask', self.inv_distrib_file, self.rng)
        else: 
            lob_state[:self.K] = sample_stationary_lob_from_file(None, self.inv_distrib_file, self.rng)
            lob_state[self.K:] = sample_stationary_lob_from_file(None, self.inv_distrib_file, self.rng)

        row = {
            'time': 0.0,
            'p_mid': self.initial_price,
            'p_ref': self.initial_price,
            'side': 'NA',
            'depth': 'NA',
            'event': 'NA',
            'redrawn': 'NA'
        }

        # Set initial queue sizes
        for i in range(self.K):
            row[f'q_bid{i+1}'] = lob_state[i]
            row[f'q_ask{i+1}'] = lob_state[self.K + i]

        cols = ['time', 'p_mid', 'p_ref', 'side', 'depth', 'event', 'redrawn'] + \
               [f'q_bid{i+1}' for i in range(self.K)][::-1] + [f'q_ask{i+1}' for i in range(self.K)] # [q_bidK, ..., q_bid1, q_ask1, ..., q_askK] format
        self.current_LOB = pd.DataFrame([row], columns=cols)

    def current_time(self):
        """
            Returns the current time of the simulation.
        """
        return self.current_LOB.iloc[-1]['time']

    def current_mid_price(self):
        """
            Returns the current mid price of the simulation.
        """
        return self.current_LOB.iloc[-1]['p_mid']

    def current_ref_price(self):
        """
            Returns the current reference price of the simulation.
        """
        return self.current_LOB.iloc[-1]['p_ref']

    # def current_queue_sizes(self):
    #     """
    #         Returns array of the current queue sizes in the format 
    #         [q_bid1, ..., q_bidK, q_ask1, ..., q_askK].
    #     """
    #     columns = [f'q_bid{i+1}' for i in range(self.K)] + [f'q_ask{i+1}' for i in range(self.K)]
    #     return self.current_LOB.iloc[-1][columns].values.astype(int)

    def find_next_trader_time(self, current_time: float):
        """
            Returns the next trader decision time (strictly) after `current_time`.
        """
        idx = np.searchsorted(self.trader_times, current_time, side='right')
        return self.trader_times[idx] if idx < len(self.trader_times) else np.inf

    def simulate_step(self):
        """
            Simulates the QRM model up to the next trader time.
            We run this code just after the agent's action.
        """
        now = self.current_time()
        next_t = self.find_next_trader_time(now)
        if next_t == np.inf:
            raise ValueError('No more trader times available.')
        
        new_LOB, _ = simulate_QRM(
            current_LOB=self.current_LOB,
            intensity_table=self.intensity_table,
            tick=self.tick,
            theta=self.theta,
            theta_reinit=self.theta_reinit,
            time_end=next_t,
            rng=self.rng,
            inv_dist_file=self.inv_distrib_file, 
            bid_ask_sym=self.bid_ask_sym
        )
        self.current_LOB = new_LOB
        return new_LOB



class MarketEnvironment:
    """
        Class for the market environment.
        This class implements the RL environment for optimal execution with market interaction.
        It uses the QueueReactiveMarketSimulator class to simulate the market and interact with it.
    """
    def __init__(
        self,
        intensity_table: IntensityTable,
        actions: list, 
        theta: float, 
        theta_reinit: float,
        tick: float,
        arrival_price: float,
        inv_distrib_file: str,
        trader_times: np.ndarray,
        initial_inventory: float,
        time_horizon: float,
        final_penalty: float,
        risk_aversion: float,
        rng: np.random.Generator,
        bid_ask_sym: bool, 
        price_offset: float, 
        price_std: float, 
        vol_offset: float,
        vol_std: float
    ):
        """
            Initializes the environment for optimal execution with market interaction.
        """
        self.intensity_table = intensity_table                  # table of estimated intensities
        self.actions = actions                                  # list of possible actions
        self.tick = tick                                        # tick size
        self.arrival_price = arrival_price                      # arrival of the implementation shortfall (IS)
        self.inv_distrib_file = inv_distrib_file                # file to sample invariant distribution
        self.trader_times = trader_times                        # trader decision times (in seconds)
        self.initial_inventory = initial_inventory              # initial inventory of the trader
        self.current_inventory = self.initial_inventory         # current inventory of the trader
        self.time_horizon = time_horizon                        # time horizon of the simulation
        self.final_penalty = final_penalty                      # penalty for not executing the entire inventory
        self.risk_aversion = risk_aversion
        self.bid_ask_sym = bid_ask_sym                          # boolean for bid-ask symmetry
        self.price_offset = price_offset                        # for normalization purposes
        self.price_std = price_std                              # for normalization purposes
        self.vol_offset = vol_offset                            # for normalization purposes
        self.vol_std = vol_std                                  # for normalization purposes
        self.current_is = 0                                     # IS every time the trader interacts with the market
        self.final_is = 0                                       # cumulative IS at the end of the episode
        self.rng = rng                                          # random number generator for reproducibility

        self.simulator = QueueReactiveMarketSimulator(
            intensity_table=self.intensity_table,
            theta=theta,
            theta_reinit=theta_reinit,
            initial_price=self.arrival_price,
            rng=self.rng,
            tick=self.tick,
            inv_distrib_file=self.inv_distrib_file,
            trader_times=self.trader_times, 
            bid_ask_sym=self.bid_ask_sym
        )

    def current_ref_price(self):
        """
            Returns the current reference price of the simulation.
        """
        return self.simulator.current_ref_price()
    
    def current_time(self):
        """
            Returns the current time of the simulation.
        """
        return self.simulator.current_time()
    

    # def current_queue_sizes(self):
    #     """
    #         Returns array of the current queue sizes in the format 
    #         [q_bid1, ..., q_bidK, q_ask1, ..., q_askK].
    #     """
    #     return self.simulator.current_queue_sizes()

    def best_quotes(self):
        """
            Extracts best quotes from the current LOB.
            Returns the best bid and ask in the format (price, size, depth).
        """
        last_lob_state = self.simulator.current_LOB.iloc[-1]
        columns = [f'q_bid{i+1}' for i in range(self.simulator.K)] + [f'q_ask{i+1}' for i in range(self.simulator.K)]
        state = last_lob_state[columns].values.astype(int)  # [q_bid1, ..., q_bidK, q_ask1, ..., q_askK] format

        # bid
        best_bid_idx = next((i+1 for i in range(self.simulator.K) if state[i] > 0), None)
        if best_bid_idx is None:
            raise ValueError('No best bid found.')
        best_bid_price = self.current_ref_price() - self.tick * (best_bid_idx - 0.5)
        best_bid_size = last_lob_state[f'q_bid{best_bid_idx}']
        best_bid = (best_bid_price, best_bid_size, best_bid_idx)

        # ask
        best_ask_idx = next((j+1 for j in range(self.simulator.K) if state[self.simulator.K + j] > 0), None)
        if best_ask_idx is None:
            raise ValueError('No best ask found.')
        best_ask_price = self.current_ref_price() + self.tick * (best_ask_idx - 0.5)
        best_ask_size = last_lob_state[f'q_ask{best_ask_idx}']
        best_ask = (best_ask_price, best_ask_size, best_ask_idx)

        return best_bid, best_ask
    
    def add_LOB(self, new_line: dict):
        """
            Adds a new event to the current LOB (trader actions).
        """
        new_df = pd.DataFrame([new_line])
        self.simulator.current_LOB = pd.concat([self.simulator.current_LOB, new_df], ignore_index=True)

    def find_next_trader_time(self, current_time: float):
        """
            Returns the next trader decision time (strictly) after `current_time`.
        """
        idx = np.searchsorted(self.trader_times, current_time, side='right')
        return self.trader_times[idx] if idx < len(self.trader_times) else np.inf


#### IMPROVEMENT NEEDED HERE
    ## ADD SEVERAL PREVIOUS STATES (input of NN, i think that previous timestamps and inventories are unecessary as they dont give additonal information
           # take only the distinct previous best quotes ?)
    ## ADD THE CURRENT INTENSITIES

    def get_state(self):
        """
            State space of the RL agent.
        """
        best_bid, best_ask = self.best_quotes()
        current_time = self.current_time()

        # pick the next trader time
        next_time = self.find_next_trader_time(current_time) 
        if next_time == np.inf: # boundary case
            assert np.abs(current_time - self.trader_times[-1]) < 1e-5, "Error in last trader time."
            time_step = self.trader_times[1] - self.trader_times[0]
            next_time = current_time + time_step
        
        return {
            'inventory': self.current_inventory,
            'time': next_time,
            'best_ask_price_1': best_ask[0],
            'best_ask_size_1': best_ask[1],
            'best_bid_size_1': best_bid[1]
        } 
    

    def state_to_vector(self, state):
        """
            Maps the state space values in, more or less, the interval [-1, 1].
        """    
        inventory = state['inventory']
        time = state['time']
        best_ask_price_1 = state['best_ask_price_1']
        best_ask_size_1 = state['best_ask_size_1']
        best_bid_size_1 = state['best_bid_size_1']
        # best_bid_price_1 = state['best_bid_price_1']

        inventory_norm = (2 * inventory / self.initial_inventory - 1)
        time_norm = (2 * time / self.time_horizon - 1)  
        best_ask_price_1_norm = (best_ask_price_1 - self.arrival_price - self.price_offset) / self.price_std
        # best_bid_price_1_norm = (best_bid_price_1 - self.arrival_price - self.price_offset) / self.price_std
        best_ask_size_1_norm = (best_ask_size_1 - self.vol_offset) / self.vol_std
        best_bid_size_1_norm = (best_bid_size_1 - self.vol_offset) / self.vol_std
        
        return np.array([
            inventory_norm, time_norm, best_ask_price_1_norm,
            best_ask_size_1_norm, best_bid_size_1_norm]) # best_bid_price_1_norm


    def reset(self):
        """
            Resets the simulation and returns initial state.
            Used at the beginning of each new episode.
        """
        self.current_inventory = self.initial_inventory
        self.simulator.rng = self.rng
        self.simulator.initialize()
        self.simulator.current_LOB = self.simulator.simulate_step()
        self.final_is = 0
        return self.get_state()
    

    def step(self, action):
        """
            Modifies the LOB according to the action taken by the agent.
            If the action is a trade through (i.e. mid-price change), 
            the LOB is updated as in Section 3.1 of Huang et al. (2015).
            Simulates the market up to the next trader time if not `done`.

        """
        if action not in self.actions:
            raise ValueError(f'Action must be in {self.actions}.')

        current_time = self.current_time()
        next_time = self.find_next_trader_time(current_time)
        if next_time == np.inf:
            raise ValueError('No more trader times.')
        
        last_state = self.simulator.current_LOB.iloc[-1].copy()
        # check available liquidity on the ask side 
        total_ask = sum([last_state[f'q_ask{i+1}'] for i in range(self.simulator.K)])
        qty_left = min(action, total_ask-1) # musn't consume all liquidity
        qty_left = min(qty_left, self.current_inventory) # can't consume more than inventory
        self.current_inventory -= qty_left

        last_state['time'] = next_time
        last_state['side'] = 'ask'
        last_state['depth'] = qty_left
        last_state['event'] = 'trader'

        reward = 0
        p_ref = self.current_ref_price()
        best_ask_idx = next((i for i in range(self.simulator.K) if last_state[f'q_ask{i+1}'] > 0), None)
        assert best_ask_idx is not None, "No best ask found."

        for depth in range(self.simulator.K):
            available = last_state[f'q_ask{depth+1}']
            take = min(qty_left, available)
            last_state[f'q_ask{depth+1}'] -= take
            qty_left -= take
            reward += (self.arrival_price - (p_ref + self.tick * (depth + 0.5))) * take
            if qty_left <= 0:
                break
        
        # Update implementation shortfall
        self.current_is = reward
        self.final_is += reward

        ################# CHANGE IN REWARD FUNCTION ###############
        ## IMPROVEMENT NEEDED HERE:
            # add a coefficent in front to normalize 
        reward -= self.current_inventory
        #########################################################
        
        new_best_ask_idx = next((i for i in range(self.simulator.K) if last_state[f'q_ask{i+1}'] > 0), None)
        assert new_best_ask_idx is not None, "No best ask found."

        # no trade through
        if new_best_ask_idx == best_ask_idx: 
            self.add_LOB(last_state)

        # trade through, modify as in Section 3.1 of Huang et al. (2015)
        else:                                
            cols = [f'q_bid{i+1}' for i in range(self.simulator.K)] + [f'q_ask{i+1}' for i in range(self.simulator.K)]
            lob_state = last_state[cols].values.astype(int).copy() 
            p_mid, ref_price, lob_state, redrawn = update_LOB(
                self.simulator.K, p_ref, lob_state, 1, 
                self.rng, self.simulator.theta, self.simulator.theta_reinit,
                self.tick, self.inv_distrib_file, self.bid_ask_sym
            )
            last_state['p_mid'] = p_mid
            last_state['p_ref'] = ref_price
            last_state['redrawn'] = redrawn

            for j in range(self.simulator.K):
                last_state[f'q_bid{j+1}'] = lob_state[j]
                last_state[f'q_ask{j+1}'] = lob_state[self.simulator.K + j]
            
            self.add_LOB(last_state)

        # Termination condition
        done = False
        if next_time >= self.time_horizon or self.current_inventory <= 0:
            done = True
            if self.current_inventory > 0:
                reward -= self.final_penalty * self.current_inventory
        
        # Simulate the market up to the next trader time
        if not done:
            self.current_LOB = self.simulator.simulate_step()

        new_state = self.get_state()
        
        return new_state, reward, done
    

    # def step_old(self, action):
    #     """
    #         Old version of the step function. Only performs action 0 or 1.
    #         Performs one step: action = 1 to sell, action = 0 to wait.
    #         Only sells if best ask has enough volume.
    #         the column 'depth' becomes the number of units traded (only for trader actions).
    #     """
    #     if action not in [0, 1]:
    #         raise ValueError('Action must be 0 or 1.')
        
    #     current_state = self.get_state()
    #     next_time = current_state['time']
    #     if next_time == np.inf:
    #         raise ValueError('No more trader times.')

    #     if action == 0 or current_state['best_ask_size_1'] == 1:
    #         last_lob_state = self.simulator.current_LOB.iloc[-1].copy()
    #         last_lob_state['time'] = next_time
    #         last_lob_state['side'] = 'ask'
    #         last_lob_state['depth'] = 0
    #         last_lob_state['event'] = 'trader'
    #         self.add_LOB(last_lob_state)
    #         reward = 0.0
    #     else:
    #         _, best_ask = self.best_quotes()
    #         best_ask_price, _, best_ask_idx = best_ask
    #         last_lob_state = self.simulator.current_LOB.iloc[-1].copy()
    #         last_lob_state[f'q_ask{best_ask_idx}'] -= 1
    #         last_lob_state['time'] = next_time
    #         last_lob_state['side'] = 'ask'
    #         last_lob_state['depth'] = 1
    #         last_lob_state['event'] = 'trader'
    #         self.add_LOB(last_lob_state)
    #         reward = self.arrival_price - best_ask_price
    #         self.current_inventory -= 1

    #     # Update implementation shortfall
    #     self.current_is = reward
    #     self.final_is += reward

    #     ############### CHANGE IN REWARD FUNCTION ###############
    #     reward -= self.current_inventory
    #     #########################################################

    #     done = False
    #     if next_time >= self.time_horizon or self.current_inventory <= 0:
    #         done = True
    #         if self.current_inventory > 0:
    #             reward -= self.final_penalty * self.current_inventory
        
    #     if not done:
    #         self.simulator.current_LOB = self.simulator.simulate_step()

    #     new_state = self.get_state()
        
    #     return new_state, reward, done





Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """
        Experience replay buffer for storing Transitions.
    """

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.rng = np.random.default_rng(0) # RNG of episodes will start at 1
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        indices = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    """
        Neural network for approximating the Q-values.
    """

    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(state_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, action_dim)
        # )
    
    def forward(self, x):
        return self.fc(x)



class DDQNAgent:
    """
        Double Deep Q-Network (DDQN) agent (see Van Hasselt et al. (2016)).
    """

    def __init__(self, state_dim, action_dim, rng, device='cpu',
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3, alpha=0.95, eps=0.01):
        
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        
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
        
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        
        self.loss_fn = nn.SmoothL1Loss()
        self.rng = rng


    def select_action(self, state):
        """ 
            ε-greedy action selection.
            `state` is supposed normalized by the function `state_to_vector`.
        """
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.action_dim)
        else:
            with torch.no_grad():
                return self.policy_net(state_tensor).argmax().item()
            

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


def load_model(agent, path):
    checkpoint = torch.load(path, weights_only=False)
    agent.policy_net.load_state_dict(checkpoint['policy_net'])
    agent.target_net.load_state_dict(checkpoint['target_net'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":

    ######### BEGINNING OF PARAMETERS #########

    # === Simulation Parameters ===
    episodes = 1
    time_horizon = 300
    initial_inventory = 75
    actions = [0, 1]
    bid_ask_sym = True       # adapt offset and std if False

    price_offset = 0.0
    price_std = 0.1
    vol_offset = 5      # 4 for asym
    vol_std = 4         # 3.5 for asym

    theta = 0.6
    theta_reinit = 0.85
    tick = 0.01
    arrival_price = 100.005
    trader_times = np.arange(0, int(time_horizon)+1, 1.0)
    final_penalty = 1.0
    risk_aversion = 0.

    # === Others ===
    state_dim = 5
    action_dim = len(actions)
    logging_every = 1
    rng_0 = np.random.default_rng(0)
    agent_type = 'DDQN'

    # === Epsilons ===
    linear_decay_epsilon = True   # this is just for wandb logging
    epsilons = np.linspace(1, 0.01, episodes)
    epsilon_start = 1.0 
    epsilon_decay = 0.995
    epsilon_end = 0.01

    # === NN hyper parameters ===
    gamma = 0.99
    learning_rate = 0.001
    alpha = 0.95
    eps = 0.01
    batch_size = 64
    memory_capacity = 2000
    target_update_freq = 10 
    burn_in = False

    # === Intensity Table ===
    file_name = 'aapl_corrected.npy'  

    folder_path_invariant_distrib = 'calibration_data/invariant_distribution/'
    inv_distrib_file = folder_path_invariant_distrib + file_name
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
        name=f"{agent_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            # Simulation Parameters
            "agent_type": agent_type,
            "episodes": episodes,
            "time_horizon": time_horizon,
            "initial_inventory": initial_inventory,
            "bid_ask_sym": bid_ask_sym,
            "price_offset": price_offset,
            "price_std": price_std,
            "vol_offset": vol_offset,
            "vol_std": vol_std,
            "theta": theta,
            "theta_reinit": theta_reinit,
            "final_penalty": final_penalty,
            "risk_aversion": risk_aversion,

            # Epsilon Parameters
            "linear_decay_epsilon": linear_decay_epsilon,
            "epsilon_start": epsilon_start,
            "epsilon_decay": epsilon_decay,
            "epsilon_end": epsilon_end,

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
    wandb.run.name = f"{agent_type}_{run_id}"

    # === Environment and Agent Initialization ===
    env = MarketEnvironment(
        intensity_table=inten_table,
        actions=actions,
        theta=theta,
        theta_reinit=theta_reinit,
        tick=tick,
        arrival_price=arrival_price,
        inv_distrib_file=inv_distrib_file,
        trader_times=trader_times,
        initial_inventory=initial_inventory,
        time_horizon=time_horizon,
        final_penalty=final_penalty,
        risk_aversion=risk_aversion,
        rng=rng_0, 
        bid_ask_sym=bid_ask_sym, 
        price_offset=price_offset,
        price_std=price_std,
        vol_offset=vol_offset,
        vol_std=vol_std
    )

    agent = DDQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        rng=rng_0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_capacity=memory_capacity,
        batch_size=batch_size,
        gamma=gamma,    
        lr=learning_rate,
        alpha=alpha, 
        eps=eps
    )

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


    try: 

        for ep in range(episodes):

            # Environment initialization
            rng = np.random.default_rng(ep+1)
            env.rng = rng
            state = env.reset() # this also passes the rng to the QRM market simulator
            state_vec = env.state_to_vector(state)

            # Agent initialization
            agent.rng = rng 
            agent.epsilon = epsilons[ep]
            # agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)

            done, ep_reward, rewards = False, 0.0, []
            actions_taken = []

            while not done:
                action = agent.select_action(state_vec) 
                next_state, reward, done = env.step(action)
                next_state_vec = env.state_to_vector(next_state)

                agent.store_transition(state_vec, action, reward, next_state_vec, done)
                agent.learn()

                state_vec = next_state_vec
                ep_reward += reward
                rewards.append(reward)
                actions_taken.append(action)

                wandb.log({
                    "Inventory Normalized": state_vec[0],
                    "Time Normalized": state_vec[1],
                    "Best Ask Price Normalized": state_vec[2],
                    "Best Ask Size Normalized": state_vec[3],
                    "Best Ask Size": next_state['best_ask_size_1'],
                    "Best Ask Price 1": next_state['best_ask_price_1'],
                    "Reward": reward, 
                    "Inventory": env.current_inventory, 
                    "Implementation Shortfall": env.current_is,
                    "Action": action
                })

            # === Update target network ===
            if ep % target_update_freq == 0 and ep > 0:
                agent.update_target_network()

            # === Print progress ===
            if ep % logging_every == 0:
                print(f"Episode {ep}/{episodes}, Reward: {ep_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

            # === End-of-episode book-keeping ===
            detailed_rewards[ep] = rewards
            final_inventories.append(env.current_inventory)
            episode_lengths.append(env.current_time())
            episodes_rewards.append(ep_reward)
            final_is.append(env.final_is)

            # === Wandb Logging ===
            wandb.log({
                "Episode": ep,
                "Final Reward": ep_reward,
                "Epsilon": agent.epsilon,
                "Final Inventory": env.current_inventory,
                "Final Implementation Shortfall": env.final_is,
                "Episode Length": env.current_time(),
                **{f"Action_{a}_count": actions_taken.count(a) for a in range(agent.action_dim)} 
            })


        # === Save weights of NNs and optimizer ===
        print("\n[INFO] Saving model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model(agent, path=f'save_model/ddqn_ep{ep}_{timestamp}.pth')
        print(f"[INFO] Model saved at episode {ep}. Exiting cleanly.")

        # === Finish wandb run ===
        wandb.finish()

    except KeyboardInterrupt:

        print("\n[INFO] Keyboard Interrupted. Saving model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model(agent, path=f'save_model/ddqn_ep{ep}_{timestamp}.pth')
        print(f"[INFO] Model saved at episode {ep}. Exiting cleanly.")