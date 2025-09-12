import numpy as np

""" 
GENERAL DESCRIPTION:

    Implementation of the benchmark trading strategies.
"""
    
class TWAPAgent:
    """
        Time-weighted average price (TWAP) agent.
    """

    def __init__(self, time_horizon, initial_inventory, trader_time_step, k=0):
        self.time_horizon = time_horizon
        self.initial_inventory = initial_inventory
        self.trader_time_step = trader_time_step
        self.k = k
        ratio = int(initial_inventory / (time_horizon / trader_time_step))
        self.actions_schedule = self.distribute_ones(int(time_horizon/trader_time_step), initial_inventory, ratio)

    @staticmethod
    def distribute_ones(n, n_0, ratio):
        """
            Uniformly distribute the n_0 shares to execute in an array of size n.
        """
        arr = ratio * np.ones(n, dtype=int)
        new_n_0 = n_0 % n
        for i in range(new_n_0):
            index = round(i * (n - 1) / (new_n_0 - 1)) if new_n_0 > 1 else n // 2
            arr[index] += 1
        assert np.sum(arr) == n_0, "TWAP Error: The number of shares in the array does not match n_0."
        return arr

    def select_action(self):
        """
            Return the number of shares to execute at step k. 
        """
        assert self.k-1 < len(self.actions_schedule), "TWAP Error: Index out of bounds."      
        return self.actions_schedule[self.k-1]
    

class BestVolumeAgent:
    """
        Take a specific action at regular time intervals.
        E.g., if modulo=2 and fixed_action=3, the agent takes action 3 at every odd steps, 
        and action 0 otherwise.
    """
    def __init__(self, fixed_action=-1, k=0, modulo=2):
        self.fixed_action = fixed_action
        self.k = k
        self.modulo = modulo

    def select_action(self):
        """
            Return the index of the action to execute at step k. 
        """
        if self.k % self.modulo == 1:
            return self.fixed_action
        else:
            return 0


class FrontLoadAgent:
    def __init__(self, fixed_action):
        self.fixed_action = fixed_action
    
    def select_action(self):
        return self.fixed_action


### --- Deprecated agents --- ###
## CHECK CODE BELOW BEFORE USING THESE AGENTS ##


class RandomAgent:
    """
        Random agent samples uniformly from the action space.
    """
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self):
        return np.random.choice(self.action_dim)
    
class ConstantAgent:
    """
        Aggressive agent: always buys two shares.
    """
    def __init__(self, fixed_action):
        self.fixed_action = fixed_action

    def select_action(self):
        return self.fixed_action
    
class BimodalAgent:
    """
        Aggressive agent: always buys two shares.
    """
    def __init__(self, k=0):
        self.k = k

    def select_action(self):
        if self.k % 2 == 1:
            return 1
        else:
            return 0
        

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
        fixed_action: int, 
        actions: list,
        security_margin: int
    ):
        self.time_horizon      = time_horizon
        self.initial_inventory = initial_inventory
        self.trader_time_step  = trader_time_step
        self.fixed_action      = fixed_action
        self.actions           = actions
        self.security_margin   = security_margin # start slightly earlier to ensure full liquidation because of QRM liquidity constraints
        self.n_steps = int(np.ceil(time_horizon / trader_time_step))


    def select_action(self, state, episode):
        
        exec_steps = int(np.ceil(self.initial_inventory / self.actions[self.fixed_action]))

        time_norm = state[1]
        curr_time = (time_norm + 1) * self.time_horizon / 2 # inverse transform of state normalization
        idx = round(curr_time / self.trader_time_step)

        if idx >= self.n_steps - exec_steps - self.security_margin: # 1 time step security margin
            return self.fixed_action
        else:
            return 0
    

class InactiveAgent:
    """
        Inactive agent that does not take any action.
    """
    def select_action(self):
        return 0


class PassiveAgent:
    """
        Passive agent: 50% no action, otherwise sample uniformly from the action space.
    """
    def __init__(self, action_dim, rng):
        self.action_dim = action_dim
        self.rng = rng

    def select_action(self):
        probs = [0.5] + [0.5 / (self.action_dim - 1)] * (self.action_dim - 1)
        return self.rng.choice([i for i in range(self.action_dim)], p=probs)
    