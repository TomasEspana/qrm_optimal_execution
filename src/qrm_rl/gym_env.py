import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
from qrm_core.intensity import IntensityTable
from .market_environment import MarketEnvironment


class QRMEnv(gym.Env):
    """
        Gym wrapper for the Queue Reactive Model market environment.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        intensity_table: IntensityTable,
        actions: list,
        theta: float,
        theta_reinit: float,
        tick: float,
        arrival_price: float,
        inv_bid_file: str,
        inv_ask_file: str,
        trader_times: np.ndarray,
        initial_inventory: int,
        time_horizon: int,
        final_penalty: float,
        risk_aversion: float,
        price_offset: float,
        price_std: float,
        vol_offset: float,
        vol_std: float,
        max_events: int,
        max_events_intra: int,
        history_size: int,
        alpha_ramp: float,
        basic_state: bool,
        state_dim: int,
        action_dim: int,
        **kwargs
    ):
        super().__init__()
        # Initialize the underlying market environment
        self._env = MarketEnvironment(
            intensity_table=intensity_table,
            actions=actions,
            theta=theta,
            theta_reinit=theta_reinit,
            tick=tick,
            arrival_price=arrival_price,
            inv_bid_file=inv_bid_file,
            inv_ask_file=inv_ask_file,
            trader_times=trader_times,
            initial_inventory=initial_inventory,
            time_horizon=time_horizon,
            final_penalty=final_penalty,
            risk_aversion=risk_aversion,
            price_offset=price_offset,
            price_std=price_std,
            vol_offset=vol_offset,
            vol_std=vol_std,
            max_events=max_events,
            max_events_intra=max_events_intra,
            history_size=history_size,
            alpha_ramp=alpha_ramp,
            basic_state=basic_state,
        )

        # Define action and observation spaces
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to initial state.
        Returns:
            observation (np.ndarray): initial state vector
        """
        state = self._env.reset()
        obs = self._env.state_to_vector(state).astype(np.float32)
        return obs, {}

    def step(self, action):
        """
            Take a step in the environment.
            Args:
                action (int): action index from Gym agent
            Returns:
                obs (np.ndarray): next state vector
                reward (float)
                done (bool)
                info (dict)
        """        
        ask_volumes = self._env.simulator.states[self._env.simulator.step - 1, self._env.simulator.K:]
        best_ask_volume = next(x for x in ask_volumes if x != 0)
        action_val = round(self._env.actions[action] * best_ask_volume)

        next_state, reward, done, executed, total_ask = self._env.step(action_val)
        obs = self._env.state_to_vector(next_state).astype(np.float32)

        # You can include additional diagnostics in info
        info = {
            "next_state": next_state,
            "executed": executed,
            "inventory": self._env.current_inventory,
            "implementation_shortfall": self._env.current_is,
            "total ask volume": total_ask
        }
        return obs, reward, done, False, info   

    def render(self, mode="human"):
        # Optional: visualize LOB or trading process
        pass

    def close(self):
        # Clean up resources
        pass


# Register the QRMEnv with Gym
register(
    id="QRM-v0",
    entry_point="qrm_rl.gym_env:QRMEnv",
)