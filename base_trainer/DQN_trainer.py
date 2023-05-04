from .base_trainer import *

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


class DQN_Trainer(nn.Module):
    def __init__(self, policy_net, target_net, memory, optimizer, env, val_env,trainConfig, device):
        super(DQN_Trainer, self).__init__()
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.optimizer = optimizer
        self.config = trainConfig
        self.env = env
        self.val_env = val_env
        self.steps_done = 0
        self.device = device
        self.update_num = 0
        self.best_reward = - float('inf')

    def select_action(self, observation):
        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.tensor(observation['real_obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
                action_logits = self.policy_net(state)
                inf_mask = torch.log(torch.tensor(observation['action_mask']))
                masked_logits = inf_mask + action_logits
                import pdb
                pdb.set_trace()
                return masked_logits.max(1)[1].view(1, 1)
        else:
            import pdb
            pdb.set_trace()
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def train(self):
        num_episodes = 600
        total_rewards = []
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

                # save model after update
                # if self.update_num % self.config.SAVE_INTERVAL == 0:
                #     self.policy_net.eval()
                #     self.target_net.eval()
                #     eval_rewards = self.validate()
                #     self.policy_net.train()
                #     self.target_net.train()
                #     self.save_model(eval_rewards)

                if done:
                    total_rewards.append(total_reward)
                    plot_rewards(total_rewards)
                    break

        print('Complete')
        plot_rewards(total_rewards, show_result= True)
        plt.savefig(os.path.join( self.config.SAVE_PATH, 'train_process.png') )

    def validate(self, val_rep = 1):
        with torch.no_grad():
            total_rewards = []
            for i in range(val_rep):

                observation = self.val_env.reset()

                done = False

                rewards = []

                while not done:
                    action = self.select_action(observation)

                    s_1, reward, done, _ = self.val_env.step(action.item())

                    s_0 = s_1

                    rewards.append(reward)

                    if done:
                        total_rewards.append(sum(rewards))

        return total_rewards

    def save_model(self, all_rewards):
        avg_reward = sum(all_rewards) / len(all_rewards)
        if avg_reward > self.best_reward:
            best_path = f"{self.config.SAVE_PATH}/best_checkpoint.pt"
            print(f"Saving the model that achived the best rewards so far into {best_path}")
            self.best_reward = avg_reward
            torch.save(self.state_dict(), best_path)


    def optimize(self):
        if len(self.memory) < self.config.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.config.BATCH_SIZE)
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
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.update_num += 1

    def update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config.TAU + target_net_state_dict[
                key] * (1 - self.config.TAU)
        self.target_net.load_state_dict(target_net_state_dict)
