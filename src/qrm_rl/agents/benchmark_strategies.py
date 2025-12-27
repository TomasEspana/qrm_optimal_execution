import numpy as np

""" 
GENERAL DESCRIPTION:

    Implementation of the benchmark trading strategies: TWAP and POPV.

    Note: 
        - TWAP select_action() returns the number of shares to execute at step k.
        - POPV selection_action() returns the index of the action to execute at step k.
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
        ratio = int(initial_inventory / (1 + time_horizon / trader_time_step))
        n = 1 + int(time_horizon/trader_time_step)
        self.actions_schedule = self.distribute_ones(n, initial_inventory, ratio)

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
        assert np.sum(arr) == n_0, "TWAPAgent Error: The number of shares in the array does not match n_0."
        return arr

    def select_action(self):
        """
            Return the number of shares to execute at step k. 
        """
        assert self.k-1 < len(self.actions_schedule), "TWAPAgent Error: Index out of bounds."      
        return self.actions_schedule[self.k-1]
    

class POPV:
    """
        Take a specific action at regular time intervals.
        E.g., if modulo=2 and fixed_action=3, the agent takes action 3 at every two steps, 
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