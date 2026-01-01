import torch

"""
    GENERAL DESCRIPTION:
        Utility function for loading RL agent models.
"""

def load_model(agent, path, test_mode=False):
    if not test_mode:
        if type(path) == str:
            checkpoint = torch.load(path, weights_only=False)
            agent.policy_net.load_state_dict(checkpoint['policy_net'])
            agent.target_net.load_state_dict(checkpoint['target_net'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            return 
    else:
        if not path:
            return
        else:
            checkpoint = torch.load(path, weights_only=False)
            agent.policy_net.load_state_dict(checkpoint['policy_net'])
            agent.target_net.load_state_dict(checkpoint['policy_net'])  
            agent.optimizer = None
            agent.policy_net.eval()
            agent.target_net.eval()
            agent.epsilon = 0.0