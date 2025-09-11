import wandb
from stable_baselines3.common.callbacks import BaseCallback

class InfoLoggerCallback(BaseCallback):
    """
        Pulls info dicts from the env and logs custom metrics to wandb.
    """

    def __init__(self, action_dim, verbose=0):

        super().__init__(verbose)

        self.action_dim = action_dim
        self.episode = 0
        self.episode_reward = 0.
        self.episode_is = 0.
        self.episode_length = 0
        self.actions = []

    def _on_step(self) -> bool:

        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.episode_length += 1
        self.episode_reward += reward
        self.episode_is += info["implementation_shortfall"]
        self.actions.append(info['action_idx'])

        # Log episode metrics
        if done:
            self.episode += 1
            log_dict = {
                "Episode": self.episode,
                "Final Reward": self.episode_reward,
                "Final Implementation Shortfall": self.episode_is, 
                "Final Inventory": info['inventory'], 
                "Episode Length": self.episode_length, 
                "Non Executed Liquidity Constraint": info["Non Executed Liquidity Constraint"], 
                **{f"Action_{a}_count": self.actions.count(a) for a in range(self.action_dim)}
            }

            # Reset episode metrics
            self.episode_length = 0
            self.episode_reward = 0.
            self.episode_is = 0.
            self.actions = []
        
        else:
            log_dict = {}
        
        # Log additional info
        log_dict.update({
            "Implementation Shortfall": info["implementation_shortfall"],
            "Inventory": info['inventory'],
            "Total Ask Volume": info['total_ask_volume'],
            "Reward": info['reward'], 
            "Risk Aversion Term in Reward": info['Risk Aversion Term in Reward'],
            "Action Index": info['action_idx'],
            "Executed": info['executed'], 
            "Mid Price": info['mid_price'], 
            "Ask Price": info['next_state'][2],
            "Inventory Normalized": info['obs'][0], 
            "Time Normalized": info['obs'][1], 
            "Best Ask Volume": info['best_ask_volume']
        })

        wandb.log(log_dict, step=self.num_timesteps)

        return True
    