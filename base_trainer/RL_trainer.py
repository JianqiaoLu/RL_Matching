from .base_trainer import *


class REINFORCE_Trainer():
    def __init__(self, policy_net, optimizer, env, RL_config, device):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.env = env
        self.device = device
        self.best_reward = -1
        self.config = RL_config
        self.steps_done = 0

    # def select_action(self, observation, random_choose = False):
    #     with torch.no_grad():
    #         if not random_choose:
    #             # sample a action by maximazing probabitliy
    #             state = torch.tensor(observation['real_obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
    #             action_logits = self.policyNet(state)
    #             action_logits *= torch.tensor(torch.tensor(observation['action_mask'],device = self.device))
    #             return action_logits.max(1)[1].view(1, 1).item()
    #         # sample a action by random
    #         else:
    #             return np.random.choice(self.env.action_space.n)
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
                return masked_logits.max(1)[1].view(1, 1)
        else:
            valid_actions = [item for item in range(self.env.offline + 1) if observation['action_mask'][item]]
            return torch.tensor([[np.random.choice(valid_actions)]], device=self.device, dtype=torch.long)

    def select_action_val(self, observation):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state = torch.tensor(observation['real_obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
            action_logits = self.policy_net(state)
            inf_mask = torch.log(torch.tensor(observation['action_mask']))
            masked_logits = inf_mask + action_logits
            return masked_logits.max(1)[1].view(1, 1)

    def discount_rewards(self, rewards, gamma=0.98):
        r = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()

    def validate(self, val_rep = 5):
        with torch.no_grad():
            total_rewards = []
            for i in range(val_rep):
                s_0 = self.env.reset()
                done = False
                rewards = []
                while not done:
                    action = self.select_action_val(s_0)
                    s_1, reward, done, _ = self.env.step(action)
                    s_0 = s_1
                    rewards.append(reward)
                    if done:
                        total_rewards.append(sum(rewards))
        return total_rewards

    def test(self, model_path, test_rep = 10):
        self.policy_net.load_state_dict(torch.load(os.path.join(model_path, 'best_checkpoint.pt')))
        self.policy_net.eval()
        rep_rewards_ratio = []
        rep_baseline_ratio = []
        for i in range(test_rep):
            s_0 = self.env.reset()
            done = False
            rewards = []
            while not done:
                action = self.select_action_val(s_0)
                s_1, reward, done, _ = self.env.step(action)
                print("online", self.env.online_type,"action", action, "reward", reward)
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
        plt.savefig(os.path.join(self.config.SAVE_PATH, 'comparison_pic.png'))

    def train(self):
        # Set up lists to hold results
        total_rewards = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_counter = 0
        ep = 0
        update_num = 0
        while ep < self.config.NUM_EPS:
            s_0 = self.env.reset()
            states = []
            rewards = []
            actions = []
            done = False
            while done == False:

                action = self.select_action(s_0).item()

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
                    if batch_counter == self.config.BATCH_SIZE:
                        self.optimizer.zero_grad()
                        state_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
                        reward_tensor = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
                        action_tensor = torch.LongTensor(np.array(batch_actions)).to(self.device)
                        logprob = torch.log(self.policy_net(state_tensor))
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
                        update_num += 1

                        if update_num  % self.config.SAVE_INTERVAL == 0:
                            self.policy_net.eval()
                            eval_rewards = self.validate()
                            self.policy_net.train()
                            self.save_model(eval_rewards)
                    ep += 1
                    print(ep)

        print('Complete')
        plot_rewards(total_rewards, show_result= True)
        plt.savefig(os.path.join(self.config.SAVE_PATH, 'train_process.png'))


    def save_model(self, all_rewards):
        avg_reward = sum(all_rewards) / len(all_rewards)
        if avg_reward > self.best_reward:
            best_path = f"{self.config.SAVE_PATH}/best_checkpoint.pt"
            print(f"Saving the model that achived the best rewards so far into {best_path}")
            self.best_reward = avg_reward
            torch.save(self.policy_net.state_dict(), best_path)


    def run_test(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()
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

