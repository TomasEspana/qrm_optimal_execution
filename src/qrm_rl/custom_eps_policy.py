from stable_baselines3.dqn.policies import DQNPolicy
import numpy as np


class CustomEpsMlpPolicy(DQNPolicy):
    def select_action(self, obs, deterministic=False):
        # Use exploration_rate from model
        eps = self.exploration_rate
        if np.random.rand() < eps:
            # Custom distribution for exploration
            action = np.random.choice(self.action_space.n, p=[0., 1.0])
            return action, None
        else:
            return super().select_action(obs, deterministic)