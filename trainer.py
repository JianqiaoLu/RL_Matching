import math
import os.path
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Transition
from algorithms.Ranking import Ranking
from algorithms.Max_matching import Max_matching

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def match_size(res):
    sum = 0
    for i in res:
        if i != -1:
            sum += 1
    return sum

class policy_estimator(nn.Module):
    def __init__(self, observation_space, action_space):
        super(policy_estimator, self).__init__()
        self.n_inputs = observation_space
        self.n_outputs = action_space

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_outputs),
            nn.Softmax(dim=-1))

    def forward(self, x):
        action_probs = self.network(torch.FloatTensor(x))
        return action_probs


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN_Trainer():
    def __init__(self, policy_net, target_net, memory, optimizer, env, trainConfig, device):
        self.policyNet = policy_net
        self.targetNet = target_net
        self.memory = memory
        self.optimizer = optimizer
        self.trainConfig = trainConfig
        self.env = env
        self.steps_done = 0
        self.device = device

    def select_action(self, observation):
        sample = random.random()
        eps_threshold = self.trainConfig.EPS_END + (self.trainConfig.EPS_START - self.trainConfig.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.trainConfig.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.tensor(observation['real_obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
                action_logits = self.policyNet(state)
                inf_mask = torch.log(torch.tensor(observation['action_mask']))
                masked_logits = inf_mask + action_logits
                return masked_logits.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def train(self):
        num_episodes = 600
        episode_rewards = []
        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            observation = self.env.reset()

            state = torch.tensor(observation['real_obs'], dtype=torch.float32, device=self.device).unsqueeze(0)

            total_reward = 0

            while (True):
                # each time one sample
                action = self.select_action(observation)

                # from action to get reward, next_state and other information
                observation, reward, done, _ = self.env.step(action.item())

                reward = torch.tensor([reward], device=self.device)

                # done = terminated or truncated

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation['real_obs'], dtype=torch.float32,
                                              device=self.device).unsqueeze(0)

                total_reward += reward

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                self.update()

                if done:
                    episode_rewards.append(total_reward)
                    plot_rewards(episode_rewards)
                    break

        print('Complete')
        plot_rewards(episode_rewards, show_result=True)
        plt.ioff()
        plt.show()

    def optimize(self):
        if len(self.memory) < self.trainConfig.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.trainConfig.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policyNet(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.trainConfig.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.targetNet(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.trainConfig.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policyNet.parameters(), 100)
        self.optimizer.step()

    def update(self):
        target_net_state_dict = self.targetNet.state_dict()
        policy_net_state_dict = self.policyNet.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.trainConfig.TAU + target_net_state_dict[
                key] * (1 - self.trainConfig.TAU)
        self.targetNet.load_state_dict(target_net_state_dict)


class REINFORCE_trainer():
    def __init__(self, policy_net, optimizer, env, RL_config, device):
        self.policyNet = policy_net
        self.optimizer = optimizer
        self.env = env
        self.device = device
        self.best_reward = -1
        self.RL_config = RL_config

    def select_action(self, observation, random_choose = False):
        with torch.no_grad():
            state = torch.tensor(observation['real_obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
            action_logits = self.policyNet(state)
            action_logits *= torch.tensor(observation['action_mask'])

            # sample a action by random
            if random_choose:
                action_logits /= torch.sum(action_logits[0])
                return np.random.choice(self.env.action_space.n, p=action_logits[0].numpy())

            # sample a action by maximazing probabitliy
            else:
                return action_logits.max(1)[1].view(1, 1).item()

    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()

    def test(self, model_path, test_rep = 100):
        self.policyNet.load_state_dict(torch.load(os.path.join(model_path,'best_checkpoint.pt')))
        self.policyNet.eval()
        rep_rewards_ratio = []
        rep_baseline_ratio = []
        for i in range(test_rep):
            s_0 = self.env.reset()
            done = False
            rewards = []
            while not done:
                action = self.select_action(s_0)
                s_1, reward, done, _ = self.env.step(action)
                s_0 = s_1
                rewards.append(reward)
                if done:
                    ranking_match = Ranking(self.env)
                    max_match = Max_matching(self.env)
                    rep_baseline_ratio.append(match_size(ranking_match)/match_size(max_match))
                    rep_rewards_ratio.append(sum(rewards)/match_size(max_match))
                    plot_two_curve(rep_rewards_ratio, rep_baseline_ratio)

        print('Complete')
        plot_two_curve(rep_rewards_ratio, rep_baseline_ratio, show_result=True)
        print("base average ratio:", sum(rep_baseline_ratio)/len(rep_baseline_ratio))
        print("rl average ratio:", sum(rep_rewards_ratio)/len(rep_rewards_ratio))
        plt.ioff()
        plt.show()

    def train(self):
        # Set up lists to hold results
        total_rewards = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_counter = 0
        ep = 0
        while ep < self.RL_config.NUM_EPS:
            s_0 = self.env.reset()
            states = []
            rewards = []
            actions = []
            done = False
            while done == False:
                action = self.select_action(s_0)
                s_1, reward, done, _ = self.env.step(action)
                states.append(s_0['real_obs'])
                rewards.append(reward)
                actions.append(action)
                s_0 = s_1
                if done:
                    batch_rewards.extend(self.discount_rewards(rewards))
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_counter += 1
                    total_rewards.append(sum(rewards))
                    plot_rewards(total_rewards)
                    # If batch is complete, update network
                    if batch_counter == self.RL_config.BATCH_SIZE:
                        self.optimizer.zero_grad()
                        state_tensor = torch.FloatTensor(np.array(batch_states))
                        reward_tensor = torch.FloatTensor(np.array(batch_rewards))
                        action_tensor = torch.LongTensor(np.array(batch_actions))
                        logprob = torch.log(self.policyNet(state_tensor))
                        selected_logprobs = reward_tensor * (
                            torch.gather(logprob, 1, action_tensor.unsqueeze(1)).squeeze())
                        loss = -selected_logprobs.mean()
                        # Calculate gradients
                        loss.backward()
                        # Apply gradients
                        self.optimizer.step()

                        batch_rewards = []
                        batch_actions = []
                        batch_states = []
                        batch_counter = 0

                    if ep % self.RL_config.SAVE_INTERVAL == 0:
                        self.save_model(total_rewards[-1])

                    ep += 1


    def save_model(self, current_total_reward):
        if current_total_reward > self.best_reward:
            best_path = f"{self.RL_config.SAVE_PATH}/best_checkpoint.pt"
            print(f"Saving the model that achived the best rewards so far into {best_path}")
            self.best_reward = current_total_reward
            torch.save(self.policyNet.state_dict(), best_path)


    def run_test(self, model_path):
        self.policyNet.load_state_dict(torch.load(model_path))
        self.policyNet.eval()
        s_0 = self.env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            action = self.select_action(s_0)
            s_1, reward, done, _ = self.env.step(action)
            states.append(s_0['real_obs'])
            rewards.append(reward)
            actions.append(action)
        return sum(rewards)


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
