import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from .network import DQNNetwork
from .replay import ReplayMemory, Transition


class DDQNAgent:
    """
        Double Deep Q-Network (DDQN) agent (see Van Hasselt et al. (2016)).
    """

    def __init__(self, state_dim=1, action_dim=1, device='cpu',
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3,
                 alpha=0.95, eps=0.01, proba_0=0.8, warmup_steps=5000):
        
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.proba_0 = proba_0
        self.warmup_steps = warmup_steps
        
        # Policy and target networks
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.lr
        )
        
        # self.optimizer = optim.RMSprop(
        #     self.policy_net.parameters(),
        #     lr=self.lr,       
        #     alpha=self.alpha,     
        #     eps=self.eps          
        #     )
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.exploration_mode = 'rl'
        
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        
        self.loss_fn = nn.SmoothL1Loss()

        # === Trading strategies ===
        self.fixed_action = 2
        self.twap_agent = None
        self.backload_agent = None


    def select_action(self, state, episode):
        """ 
            ε-greedy action selection.
            `state` is supposed normalized by the function `state_to_vector`.
        """
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.exploration_mode == 'rl':
            if np.random.random() < self.epsilon:
                actions = [i for i in range(self.action_dim)]
                probs = [self.proba_0] + [(1-self.proba_0) / (self.action_dim - 1)] * (self.action_dim - 1)
                return np.random.choice(actions, p=probs)
            else:
                with torch.no_grad():
                    return self.policy_net(state_tensor).argmax().item()
        
        elif self.exploration_mode == 'front_load':
            return self.fixed_action
        
        elif self.exploration_mode == 'back_load':
            self.backload_agent.fixed_action = self.fixed_action
            self.backload_agent.exec_steps = int(np.ceil(self.backload_agent.initial_inventory / self.fixed_action))
            return self.backload_agent.select_action(state, episode)
        
        elif self.exploration_mode == 'twap':
            return self.twap_agent.select_action(state, episode)
        
        else:
            raise ValueError(f"Unknown exploration mode: {self.exploration_mode}")
            

    def store_transition(self, state, action, reward, next_state, done):
        """ 
            Store transition in the replay memory.
            `state` and `next_state` is supposed normalized by the function `state_to_vector`.
        """
        self.memory.push(
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done
        )


    def learn(self):
        """
            Sample a batch of transitions from the replay memory and update the policy network.
        """
        if len(self.memory) < self.warmup_steps:
            return {}
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        all_q_values = self.policy_net(state_batch)
        current_q_values = all_q_values.gather(1, action_batch)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            max_next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)
        
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Log total gradient norm
        total_norm = torch.norm(
            torch.stack([param.grad.norm() for param in self.policy_net.parameters() if param.grad is not None])
        )

        wandb_dic = {
            "Loss - TD Error": loss.item(),
            "Policy Q Values - Mean": all_q_values.mean().item(),
            "Policy Q Values - Std": all_q_values.std().item(),
            "grad_norm/total": total_norm.item()
        }
        for i in range(self.action_dim):
            wandb_dic[f"Policy Q Value - Action {i} - Mean"] = all_q_values[:, i].mean().item()
            wandb_dic[f"Policy Q Value - Action {i} - Std"] = all_q_values[:, i].std().item()

        # # Log gradient norm for each layer
        # layer_grads = defaultdict(list)
        # for name, param in self.policy_net.named_parameters():
        #     if param.grad is not None:
        #         layer_name = name.split('.')[0]  # e.g., 'fc1' from 'fc1.weight'
        #         layer_grads[layer_name].append(param.grad.view(-1))

        # for layer, grads in layer_grads.items():
        #     total_norm = torch.norm(torch.cat(grads)).item()
        #     wandb.log({f"grad_norm/{layer}": total_norm})

        ### --
        self.optimizer.step()

        return wandb_dic


    def update_target_network(self):
        """
            Update the target network with the weights of the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())