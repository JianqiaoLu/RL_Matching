import math
import torch.nn as nn
import torch.nn.functional as F
from utils import Transition
import torch
import random
import math
import os.path
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils import Transition
from algorithms.Ranking import Ranking
from algorithms.Max_matching import Max_matching
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# set up matplotlib
plt.ion()

class policy_estimator(nn.Module):
    def __init__(self, observation_space, action_space):
        super(policy_estimator, self).__init__()
        self.n_inputs = observation_space
        self.n_outputs = action_space
        hidden_states = 1024

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, self.n_outputs),
            nn.Softmax(dim=-1))

    def forward(self, x):
        action_probs = self.network(torch.FloatTensor(x))
        return action_probs
def plot_durations(episode_durations, show_result=False):
    plt.figure(1)

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_rewards(episode_rewards, show_result=False):
    plt.figure(1)

    rewards = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards.numpy())
    # Take 100 episode averages and plot them too
    # if len(rewards) >= 100:
    #     means = rewards.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_two_curve(episode_rewards, episode_baseline,show_result=False):
    plt.figure(1)

    # rewards = torch.tensor(episode_rewards, dtype=torch.float)
    # baseline = torch.tensor(episode_baseline, dtype=torch.float)
    if show_result:
        plt.clf()
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(list(zip(episode_rewards, episode_baseline)), label=["rewards","baseline"])
    if show_result:
        plt.legend()
    # Take 100 episode averages and plot them too
    # if len(rewards) >= 100:
    #     means = rewards.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def match_size(res):
    sum = 0
    for i in res:
        if i != -1:
            sum += 1
    return sum
