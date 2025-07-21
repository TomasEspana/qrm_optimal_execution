import torch


def save_model(agent, path):
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict()
    }, path)


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



### --- Not used at the moment --- ###

def generate_boundary_episodes(env, strategy="sell_first", n_pretrain_paths=100):

    episodes = []
    
    for _ in range(n_pretrain_paths):

        state = env.reset()
        # state_vec = state_to_vector(state)
        
        (state)
        transitions = []
        done = False

        while not done:

            if strategy == "sell_first":
                action = 4 
            elif strategy == "sell_last":
                if env.current_time < env.time_horizon - 25:
                    action = 0
                else:
                    action = 4
            elif strategy == "do_nothing":
                action = 0
            
            next_state, reward, done = env.step(action)
            # next_state_vec = state_to_vector(next_state)
            # transitions.append((state_vec, action, reward, next_state_vec, done))

        episodes.extend(transitions)

    return episodes
        