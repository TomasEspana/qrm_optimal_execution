import pickle 
from src.qrm_rl.configs.config import load_config
from src.qrm_rl.runner import RLRunner
from src.qrm_rl.agents.benchmark_strategies import TWAPAgent, BackLoadAgent, FrontLoadAgent, RandomAgent

if __name__ == "__main__":


    ### Specify the run id of the trained model to test. ###
    ### ----------------------###
    train_run_id = 't40nwadb'
    ### ----------------------###

    config = load_config()
    config['mode'] = 'test'

    runner = RLRunner(config, load_model_path=f'save_model/ddqn_{train_run_id}.pth')

    th = runner.cfg['time_horizon']
    ii = runner.cfg['initial_inventory']
    tts = runner.cfg['trader_time_step']
    actions = runner.cfg['actions']
    max_action = max(actions)
    final_is = {}


    ### === DDQN Agent Testing === ###
    dic, run_id = runner.run()
    final_is['DDQN'] = dic['final_is']
    with open(f'data_wandb/dictionaries/ddqn_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === TWAP Agent Testing === ###
    runner = RLRunner(config)
    agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
    # # avoid remaining inventory because of the QRM liquidity constraints
    # actions = agent.actions
    # actions[-1] += 1
    # agent.actions = actions

    runner.agent = agent
    dic, run_id = runner.run()
    final_is['TWAP'] = dic['final_is']
    with open(f'data_wandb/dictionaries/twap_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === Back Load Agent Testing === ###
    runner = RLRunner(config)
    agent = BackLoadAgent(time_horizon=th, initial_inventory=ii, 
                            trader_time_step=tts, fixed_action=max_action, security_margin=runner.cfg['exec_security_margin'])

    runner.agent = agent
    dic, run_id = runner.run()
    final_is['Back Load'] = dic['final_is']
    with open(f'data_wandb/dictionaries/back_load_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === Front Load Agent Testing === ###
    runner = RLRunner(config)
    agent = FrontLoadAgent(fixed_action=max_action)
    runner.agent = agent
    dic, run_id = runner.run()
    final_is['Front Load'] = dic['final_is']
    with open(f'data_wandb/dictionaries/front_load_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === Front Load Agent Testing === ###
    runner = RLRunner(config)
    agent = FrontLoadAgent(fixed_action=actions[1])
    runner.agent = agent
    dic, run_id = runner.run()
    final_is['Front Load - 1'] = dic['final_is']
    with open(f'data_wandb/dictionaries/front_load_1_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)

    ### === Random Agent Testing === ###
    runner = RLRunner(config)
    agent = RandomAgent(action_dim=runner.cfg['action_dim'])
    runner.agent = agent
    dic, run_id = runner.run()
    final_is['Random'] = dic['final_is']
    with open(f'data_wandb/dictionaries/random_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f)