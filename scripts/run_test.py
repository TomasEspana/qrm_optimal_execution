import pickle 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) 

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner
from qrm_rl.agents.benchmark_strategies import TWAPAgent, BackLoadAgent, FrontLoadAgent, RandomAgent, ConstantAgent, BimodalAgent, BestVolumeAgent

if __name__ == "__main__":

    config = load_config()
    ### Specify the run id of the trained model to test. ###
    ### ----------------------###
    train_run_id = 'glbpbi4x'
    config['test_save_memory'] = True
    ### ----------------------###

    config['mode'] = 'test'
    config['episodes'] = 20

    runner = RLRunner(config)

    th = runner.cfg['time_horizon']
    ii = runner.cfg['initial_inventory']
    tts = runner.cfg['trader_time_step']
    actions = runner.cfg['actions']
    max_action = max(actions)


    ## === Best Volume - Agent Testing === ###
    runner = RLRunner(config)
    agent = BestVolumeAgent(fixed_action=-1, modulo=2)
    runner.agent = agent
    dic, run_id = runner.run()
    with open(f'data_wandb/dictionaries/best_volume_{train_run_id}.pkl', 'wb') as f:
        pickle.dump(dic, f) 
    
    # # ### === Front Load Agent Testing === ###
    # runner = RLRunner(config)
    # agent = FrontLoadAgent(fixed_action=-1)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/front_load_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === DDQN Agent Testing === ###
    # runner = RLRunner(config, load_model_path=f'save_model/ddqn_{train_run_id}.zip')
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/ddqn_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === TWAP Agent Testing === ###
    # runner = RLRunner(config)
    # agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/twap_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)



    # ## === Bimodal - Agent Testing === ###
    # runner = RLRunner(config)
    # agent = BimodalAgent()
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/bimodal_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f) 

    # ### === Constant 0 - Agent Testing === ###
    # runner = RLRunner(config)
    # agent = ConstantAgent(fixed_action=0)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/constant_0_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Constant Agent Testing === ###
    # runner = RLRunner(config)
    # agent = ConstantAgent(fixed_action=1)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/constant_1_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Constant Agent Testing === ###
    # runner = RLRunner(config)
    # agent = ConstantAgent(fixed_action=2)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/constant_2_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Constant Agent Testing === ###
    # runner = RLRunner(config)
    # agent = ConstantAgent(fixed_action=3)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/constant_3_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)









    # ### === DDQN Agent Testing === ###
    # runner = RLRunner(config, load_model_path=f'save_model/ddqn_{train_run_id}.zip')
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/ddqn_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === TWAP Agent Testing === ###
    # runner = RLRunner(config)
    # agent = TWAPAgent(time_horizon=th, initial_inventory=ii, trader_time_step=tts)
    # # # avoid remaining inventory because of the QRM liquidity constraints
    # # actions_ag = agent.actions
    # # actions_ag[-1] += 1
    # # agent.actions = actions_ag

    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/twap_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Back Load Agent Testing === ###
    # runner = RLRunner(config)
    # agent = BackLoadAgent(time_horizon=th, initial_inventory=ii, 
    #                         trader_time_step=tts, fixed_action=-1, actions=actions, security_margin=0)

    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/back_load_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Front Load Agent Testing === ###
    # runner = RLRunner(config)
    # agent = FrontLoadAgent(fixed_action=-1)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/front_load_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Front Load - 1 - Agent Testing === ###
    # runner = RLRunner(config)
    # agent = FrontLoadAgent(fixed_action=1)
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/front_load_1_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === Random Agent Testing === ###
    # runner = RLRunner(config)
    # agent = RandomAgent(action_dim=len(actions))
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/random_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)

    # ### === BiModal Agent Testing === ###
    # runner = RLRunner(config)
    # agent = BimodalAgent()
    # runner.agent = agent
    # dic, run_id = runner.run()
    # with open(f'data_wandb/dictionaries/bimodal_{train_run_id}.pkl', 'wb') as f:
    #     pickle.dump(dic, f)