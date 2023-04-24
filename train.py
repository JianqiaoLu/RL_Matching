import gymnasium as gym
import torch.optim as optim
from itertools import count
from memory import ReplayMemory
from trainer import *
from utils import TrainConfig, RL_TrainConfig
from online_matching_environment import BipartiteMatchingActionMaskGymEnvironment
from algorithms import *

# set up matplotlib
plt.ion()

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
)

RL_config = RL_TrainConfig(
    BATCH_SIZE=128,
    SAVE_PATH='/Users/jianqiaolu/discuss with zhiyi/rl/RL_model',
    SAVE_INTERVAL=10,
    LR=1e-4,
)
env_config = {
    "bag_capacity": 100,
    'item_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'item_probabilities': [0, 0, 0, 1 / 3, 0, 0, 0, 0, 2 / 3],  # linear waste
    # 'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
    # 'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33], #perfect pack
    'time_horizon': 1000
}




if __name__ == '__main__':
    # set env
    # env = gym.make("CartPole-v1")
    # env = BinPackingActionMaskGymEnvironment()
    env = BipartiteMatchingActionMaskGymEnvironment(file_name="real_graph/socfb-Caltech36/socfb-Caltech36.txt")
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
    optimizer = optim.AdamW(policy_net.parameters(), lr=DQN_config.LR, amsgrad=True)
    memory = ReplayMemory(10000)
    dqn_trainer = DQN_Trainer(policy_net, target_net, memory, optimizer, env, DQN_config, device)
    # dqn_trainer.train()

    # REINFORCE trainer model
    policy_net2 = policy_estimator(n_observations, n_actions)
    optimizer = optim.AdamW(policy_net2.parameters(), lr=RL_config.LR, amsgrad=True)

    rl_trainer = REINFORCE_trainer(policy_net2, optimizer, env, RL_config, device)
    rl_trainer.train()
