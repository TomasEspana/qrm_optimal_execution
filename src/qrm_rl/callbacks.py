import wandb
from stable_baselines3.common.callbacks import BaseCallback

class InfoLoggerCallback(BaseCallback):
    """
        Pulls info dicts from the env and logs custom metrics to wandb.
        Works with VecEnv; infos is a list (one per env).
    """
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info:
                continue
            if "implementation_shortfall" in info:
                wandb.log({
                    "Implementation Shortfall": info["implementation_shortfall"],
                    "Inventory": info.get("inventory", np.nan),
                }, step=self.num_timesteps)
        return True