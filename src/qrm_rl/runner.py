import wandb
import numpy as np
import torch
from numba import njit
from contextlib import nullcontext



exploration_mode_dic = {'rl': int(0), 'front_load': int(1), 'back_load': int(2), 'twap': int(3)}

class RLRunner:
    def __init__(self, config, load_model_path=None):
        # Unpack config
        self.cfg = config
        self.mode = config['mode']
        self.episodes = config['episodes']
        self.exec_security_margin = config['exec_security_margin']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.unif_deter_strats = config['unif_deter_strats']
        self.prop_greedy_eps = config['prop_greedy_eps']
        self.prop_deter_strats = config['prop_deter_strats'] 
        self.agent = None
        self.agent_name_map = {
            DDQNAgent: 'ddqn', 
            InactiveAgent: 'inactive', 
            RandomAgent: 'random',
            PassiveAgent: 'passive',
            TWAPAgent: 'twap', 
            BackLoadAgent: 'back_load',
            FrontLoadAgent: 'front_load'
        }

        # Seeds
        np.random.seed(config['seed'])
        @njit
        def _init_numba(seed): np.random.seed(seed)
        _init_numba(config['seed'])

        # WandB init
        wandb.init(
            project="QRM_RL_Agent",
            name=f"{self.mode}",
            config=config
        )

        # Build intensity table
        inten_arr = np.load(config['folder_path_intensity_table'] + config['file_name'])
        K, Qp1, *_ = inten_arr.shape
        inten_table = IntensityTable(max_depth=K, max_queue=Qp1-1)
        inten_table._data = inten_arr

        # Environment
        self.env = MarketEnvironment(
            intensity_table=inten_table,
            actions=config['actions'],
            theta=config['theta'],
            theta_reinit=config['theta_reinit'],
            tick=config['tick'],
            arrival_price=config['arrival_price'],
            inv_bid_file=config['folder_path_invariant'] + config['file_name_bid'],
            inv_ask_file=config['folder_path_invariant'] + config['file_name_ask'],
            trader_times=config['trader_times'],
            initial_inventory=config['initial_inventory'],
            time_horizon=config['time_horizon'],
            final_penalty=config['final_penalty'],
            risk_aversion=config['risk_aversion'],
            price_offset=config['price_offset'],
            price_std=config['price_std'],
            vol_offset=config['vol_offset'],
            vol_std=config['vol_std'],
            max_events=config['max_events'],
            max_events_intra=config['max_events_intra'], 
            history_size=config['history_size'], 
            alpha_ramp=config['alpha_ramp']
        )

        # Agent
        self.agent = DDQNAgent(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            device=self.device,
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay'],
            memory_capacity=config['memory_capacity'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            lr=config['learning_rate'],
            alpha=config['alpha'],
            eps=config['eps'],
            proba_0=config['proba_0']
        )

        if self.mode == 'train':
            self.agent.twap_agent = TWAPAgent(
                time_horizon=config['time_horizon'],
                initial_inventory=config['initial_inventory'],
                trader_time_step=config['trader_time_step']
            )
            self.agent.backload_agent = BackLoadAgent(
                time_horizon=config['time_horizon'],
                initial_inventory=config['initial_inventory'],
                trader_time_step=config['trader_time_step'], 
                fixed_action=1
            )

        # Load weights
        test_mode = (self.mode == 'test')
        load_model(self.agent, load_model_path, test_mode=test_mode)

    def _update_epsilon(self, ep):
        E = self.episodes
        bs = int(self.prop_deter_strats * E)
        ge = int(self.prop_greedy_eps * E)

        decay_steps = ge - bs
        eps_start, eps_end = self.cfg['epsilon_start'], self.cfg['epsilon_end']
        a = (eps_end - eps_start) / decay_steps
        b = eps_start - bs * a

        if ep < bs:
            self.agent.exploration_mode = np.random.choice(
                ['front_load','twap','back_load'], p=[0.3,0.4,0.3]
            )
            self.agent.epsilon = 0.0
            self.agent.fixed_action = np.random.choice(
                [a for a in range(1, self.agent.action_dim)]
            )
        elif ep < ge:
            self.agent.exploration_mode = 'rl'
            self.agent.epsilon = a * ep + b
        else:
            self.agent.exploration_mode = 'rl'
            self.agent.epsilon = self.cfg['epsilon_end']

    def _update_epsilon_unif(self, ep):
        """ 
            the deterministic strategies are uniformly distributed over the entire exploration phase.
        """
        E = self.episodes
        bs = int(self.prop_deter_strats * E)
        ge = int(self.prop_greedy_eps * E)

        if bs > 0:
            det_indices = set(int(np.floor(i * E / bs)) for i in range(bs))
        else:
            det_indices = set()

        eps_start, eps_end = self.cfg['epsilon_start'], self.cfg['epsilon_end']
        slope = (eps_end - eps_start) / float(ge)
        intercept = eps_start

        if ep in det_indices:
            self.agent.exploration_mode = np.random.choice(
                ['front_load', 'twap', 'back_load'], p=[0.3, 0.4, 0.3]
            )
            self.agent.epsilon = 0.0
            self.agent.fixed_action = np.random.choice(
                [a for a in range(1, self.agent.action_dim)]
            )
        elif ep < ge:
            self.agent.exploration_mode = 'rl'
            self.agent.epsilon = max(eps_end, slope * ep + intercept)
        else:
            self.agent.exploration_mode = 'rl'
            self.agent.epsilon = self.cfg['epsilon_end']


    def _log_step(self, state_vec, next_state, reward, action, dic, step):
        d = {
            "Inventory Normalized": state_vec[0],
            "Time Normalized": state_vec[1],
            "Best Ask Price Normalized": state_vec[4],
            "Best Ask Size Normalized": state_vec[5],
            "Best Ask Size": next_state[5],
            "Best Ask Price 1": next_state[4],
            "Reward": reward, 
            "Risk Aversion Term in Reward": self.env.risk_aversion_term,
            "Inventory": self.env.current_inventory, 
            "Implementation Shortfall": self.env.current_is,
            "Action": action,
            "Mid Price": self.env.current_mid_price()
        }
        d.update(dic)
        wandb.log(d, step=step)

    def run(self):
        agent_type = self.agent_name_map.get(type(self.agent), 'Unknown')
        self.run_id = wandb.run.id
        train_mode = (self.mode == 'train')
        if train_mode:
            wandb.run.name = f"{agent_type}_{self.run_id}"
        else:
            wandb.run.name = f"{agent_type}_test_{self.run_id}"

        if not train_mode:
            final_is = []
            mid_prices = {}
            actions_taken = {}
            executed_dic = {}

        step_count = 0
        for ep in range(self.episodes):

            state = self.env.reset()
            state_vec = self.env.state_to_vector(state)
            if train_mode:
                if not self.unif_deter_strats:
                    self._update_epsilon(ep)
                else:
                    self._update_epsilon_unif(ep)

            ctx = torch.no_grad() if not train_mode else nullcontext()
            with ctx:
                done, ep_reward, actions, executed = False, 0.0, [], []
                k = 1
                while not done:

                    if train_mode:

                        action = self.agent.select_action(state_vec, ep)
                        nxt, reward, done, exec = self.env.step(action)
                        nxt_vec = self.env.state_to_vector(nxt)
                        
                        self.agent.store_transition(state_vec, action, reward, nxt_vec, done)
                        wandb_dic = self.agent.learn()
                        # if step_count % self.cfg['target_update_freq'] == 0:
                        #     self.agent.update_target_network()

                    else:
                        current_inventory = self.env.current_inventory
                        t_left = np.ceil(current_inventory / max(self.env.actions)) + self.exec_security_margin 
                        if len(self.env.trader_times) - k > t_left:
                            action = self.agent.select_action(state_vec, ep)
                        else:
                            action = max(self.env.actions)
                        nxt, reward, done, exec = self.env.step(action)
                        nxt_vec = self.env.state_to_vector(nxt)
                        wandb_dic = {}

                    state_vec = nxt_vec
                    ep_reward += reward
                    actions.append(action)
                    executed.append(exec)
                    self._log_step(state_vec, nxt, reward, action, wandb_dic, step_count)
                    step_count += 1
                    k += 1
                
            if train_mode and ep % self.cfg['target_update_freq'] == 0 and ep > 0:
                self.agent.update_target_network()

            # end of episode logging
            summary = {
                "Episode": ep,
                "Final Reward": ep_reward,
                "Final Inventory": self.env.current_inventory,
                "Final Implementation Shortfall": self.env.final_is,
                "Episode Length": self.env.current_time(),
                **{f"Action_{a}_count": actions.count(a) for a in range(self.cfg['action_dim'])}, 
                "Non Executed Liquidity Constraint": self.env.non_executed_liquidity_constraint, 
            }
            if train_mode:
                summary['Epsilon'] = self.agent.epsilon
                summary['Exploration Mode'] = exploration_mode_dic[self.agent.exploration_mode]
                summary['Fixed Action'] = self.agent.fixed_action
            
            wandb.log(summary, step=step_count-1)

            # === Print progress ===
            if ep % self.cfg['logging_every'] == 0:
                print(f"[{self.mode.upper()}][{ep}/{self.episodes}]  Reward={ep_reward:.2f}") #  Eps={self.agent.epsilon:.3f}")

            # === End-of-episode book-keeping ===
            if not train_mode:
                final_is.append(self.env.final_is)
                mid_prices[ep] = self.env.simulator.p_mids[:self.env.simulator.step]
                actions_taken[ep] = actions
                executed_dic[ep] = executed

        # save model if training
        if train_mode:
            save_model(self.agent, f"save_model/{self.agent_name_map[type(self.agent)]}_{self.run_id}.pth")

        wandb.finish()

        if not train_mode:
            dic = {
                'final_is': final_is, 
                'mid_prices': mid_prices,
                'actions': actions_taken, 
                'executed': executed_dic
            }
            return dic, self.run_id
