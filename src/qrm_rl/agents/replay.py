from collections import deque, namedtuple
import numpy as np

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
        """
            Always sample the current state.
        """
        latest_transition = self.memory[-1]
        indices = np.random.choice(len(self.memory)-1, size=batch_size-1, replace=False)
        batch = [self.memory[i] for i in indices]
        batch.append(latest_transition)
        return batch
    
    def __len__(self):
        return len(self.memory)