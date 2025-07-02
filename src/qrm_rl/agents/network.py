import torch.nn as nn

class DQNNetwork(nn.Module):
    """
        Neural network for approximating the Q-values.
    """

    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        ### --- Macri et al. NN ---
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 30),
            nn.LeakyReLU(),
            nn.Linear(30, action_dim)
        )
        ### --- Ning et al. NN ---
        # self.fc = nn.Sequential(
        #     nn.Linear(state_dim, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, action_dim)
        # )
        ### --- NN 1 ---- 
        # self.fc = nn.Sequential(
        #     nn.Linear(state_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, action_dim)
        # )
        ### --- NN 2 ----
        # self.fc = nn.Sequential(
        #     nn.Linear(state_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, action_dim)
        # )
    
    def forward(self, x):
        return self.fc(x)
