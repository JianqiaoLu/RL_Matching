import torch.optim as optim
from base_trainer.RL_trainer import REINFORCE_Trainer
from base_trainer.DQN_trainer import DQN_Trainer, DQN
from utils import TrainConfig, RL_TrainConfig
from online_matching_environment import BipartiteMatchingActionMaskGymEnvironment, StochasticBipartiteMatchingActionMaskGymEnvironment, BipartiteMatchingGymEnvironment_UpperTriangle, BipartiteMatchingActionMaskGymEnvironment_UpperTriangle
from memory import ReplayMemory
import torch
from base_trainer.base_trainer import policy_estimator


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



DQN_config = TrainConfig(
    BATCH_SIZE=128,
    GAMMA=0.99,
    EPS_START=0.9,
    EPS_END=0.05,
    EPS_DECAY=1000,
    TAU=0.005,
    LR=1e-4,
    SAVE_INTERVAL=1,
    SAVE_PATH='DQN_RL_model',
)

RL_config = RL_TrainConfig(
    BATCH_SIZE=10,
    SAVE_PATH='/Users/jianqiaolu/discuss with zhiyi/rl/bipartite_matching/RL_model',
    SAVE_INTERVAL=1,
    LR=1e-4,
    NUM_EPS = 3000,
    EPS_START=0.9,
    EPS_END=0.05,
    EPS_DECAY=1000,
)
env_configs= {
    'offline': 100,
    'online': 100,
    'edges': [],
    'time_horizon': 100,
}


if __name__ == '__main__':
    # set env
    # env = gym.make("CartPole-v1")
    # env = BinPackingActionMaskGymEnvironment()
    # env = BipartiteMatchingActionMaskGymEnvironment_UpperTriangle(env_config = env_configs)
    # env = BipartiteMatchingGymEnvironment_UpperTriangle(file_name="real_graph/socfb-Caltech36/socfb-Caltech36.txt")
    env = BipartiteMatchingActionMaskGymEnvironment(file_name="real_graph/socfb-Caltech36/socfb-Caltech36.txt")
    # env = BipartiteMatchingActionMaskGymEnvironment(file_name="real_graph/lp_blend/lp_blend.mtx")
    # val_env = BipartiteMatchingActionMaskGymEnvironment(file_name="real_graph/lp_blend/lp_blend.mtx")
    # env = StochasticBipartiteMatchingActionMaskGymEnvironment(file_name='real_graph/socfb-Caltech36/socfb-Caltech36.txt')
    # env = StochasticBipartiteMatchingActionMaskGymEnvironment(file_name= "real_graph/lp_blend/lp_blend.mtx")
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state = env.reset()
    # state = env.reset()



    # DQN_trainer model

    # make experience for getting reward
    n_observations = len(state['real_obs'])
    # n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # only set optimizer for policy_net
    # optimizer = optim.AdamW(policy_net.parameters(), lr=DQN_config.LR, amsgrad=True)
    # memory = ReplayMemory(10000)
    # dqn_trainer = DQN_Trainer(policy_net, target_net, memory, optimizer, env, val_env,DQN_config, device)
    # dqn_trainer.train()


    # REINFORCE trainer model
    policy_net2 = policy_estimator(n_observations, n_actions)
    optimizer = optim.AdamW(policy_net2.parameters(), lr=RL_config.LR, amsgrad=True)

    rl_trainer = REINFORCE_Trainer(policy_net2, optimizer, env, RL_config, device)
    # rl_trainer.train()
    rl_trainer.test(model_path=rl_trainer.config.SAVE_PATH)
