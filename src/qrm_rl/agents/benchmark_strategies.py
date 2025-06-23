import numpy as np

""" 
    Benchmark trading strategies.
"""
    
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
    
class RandomAgent:
    """
        Random agent samples uniformly from the action space.
    """
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state, episode):
        return np.random.choice(self.action_dim)
    


### --- Deprecated agents --- ###
## CHECK CODE BELOW BEFORE USING THESE AGENTS ##

class InactiveAgent:
    """
        Inactive agent that does not take any action.
    """
    def select_action(self, state, episode):
        return 0

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