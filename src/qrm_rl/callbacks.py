import wandb
from stable_baselines3.common.callbacks import BaseCallback

class InfoLoggerCallback(BaseCallback):
    """
        Pulls info dicts from the env and logs custom metrics to wandb.
    """

    def __init__(self, action_dim, rl_model, verbose=0):

        super().__init__(verbose)

        self.action_dim = action_dim
        self.episode = 0
        self.rl_model = rl_model
        self.episode_reward = 0.
        self.episode_is = 0.
        self.episode_length = 0
        self.actions = []
        self.loss_sum = 0.
        self.metrics_buffer = {}


    def _on_step(self) -> bool:

        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.episode_length += 1
        self.episode_reward += reward
        self.episode_is += info["implementation_shortfall"]
        self.actions.append(info['action_idx'])
        name_to_val = getattr(self.model.logger, "name_to_value", {})
        if "train/loss" in name_to_val:
            self.loss_sum += float(name_to_val["train/loss"])

        # Log episode metrics
        if done:
            self.episode += 1
            log_dict = {
                "Episode": self.episode,
                'Mean TD Loss': self.loss_sum / self.episode_length if self.episode_length > 0 else 0.,
                "Final Reward": self.episode_reward,
                "Final Implementation Shortfall": self.episode_is, 
                "Final Inventory": info['inventory'], 
                "Episode Length": self.episode_length, 
                "Non Executed Liquidity Constraint": info["Non Executed Liquidity Constraint"], 
                **{f"Action_{a}_count": self.actions.count(a) for a in range(self.action_dim)}, 
                "Final Penalty": info['final_penalty_coeff'] * info['inventory'],
            }

            # Reset EPISODE metrics
            self.episode_length = 0
            self.episode_reward = 0.
            self.episode_is = 0.
            self.actions = []
            self.loss_sum = 0.
        
        else:
            log_dict = {}
        
        # Log STEP metrics
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

        keys = [
            "train/value_loss",
            "train/loss",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/learning_rate",
            "train/explained_variance",
        ]
        payload = {k: float(v) for k, v in name_to_val.items() if k in keys}

        # DQN TD loss may also appear here (aggregate over last update)
        if "train/loss" in name_to_val:
            payload["train/td_loss"] = float(name_to_val["train/loss"])
        if payload:
            wandb.log(payload, step=self.num_timesteps)

        wandb.log(log_dict, step=self.num_timesteps)

        return True