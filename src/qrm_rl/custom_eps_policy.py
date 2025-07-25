from stable_baselines3.dqn.policies import DQNPolicy
import numpy as np


class CustomEpsMlpPolicy(DQNPolicy):
    def _predict(self, observation, deterministic=False):
        # Use exploration rate injected by callback
        eps = getattr(self, "exploration_rate", 0.1)

        if np.random.rand() < eps:
            # Custom random action distribution
            action = np.random.choice(self.action_space.n, p=[0.8, 0.2])
        else:
            q_values = self.q_net(observation)
            action = q_values.argmax(dim=1).cpu().numpy()
        return action
