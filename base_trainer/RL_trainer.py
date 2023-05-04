from .base_trainer import *


class REINFORCE_Trainer():
    def __init__(self, policy_net, optimizer, env, RL_config, device):
        self.policyNet = policy_net
        self.optimizer = optimizer
        self.env = env
        self.device = device
        self.best_reward = -1
        self.config = RL_config

    def select_action(self, observation, random_choose = False):
        with torch.no_grad():
            state = torch.tensor(observation['real_obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
            action_logits = self.policyNet(state)
            action_logits *= torch.tensor(torch.tensor(observation['action_mask'],device = self.device))

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

    def validate(self, val_rep = 5):
        with torch.no_grad():
            total_rewards = []
            for i in range(val_rep):
                s_0 = self.env.reset()
                done = False
                rewards = []
                while not done:
                    action = self.select_action(s_0)
                    s_1, reward, done, _ = self.env.step(action)
                    s_0 = s_1
                    rewards.append(reward)
                    if done:
                        total_rewards.append(sum(rewards))
        return total_rewards

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
                    if batch_counter == self.config.BATCH_SIZE:
                        self.optimizer.zero_grad()
                        state_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
                        reward_tensor = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
                        action_tensor = torch.LongTensor(np.array(batch_actions)).to(self.device)
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
                        update_num += 1

                        if update_num  % self.config.SAVE_INTERVAL == 0:
                            self.policyNet.eval()
                            eval_rewards = self.validate()
                            self.policyNet.train()
                            self.save_model(eval_rewards)

                    ep += 1

        print('Complete')
        plot_rewards(total_rewards, show_result= True)
        plt.savefig(os.path.join(self.config.SAVE_PATH, 'train_process.png'))


    def save_model(self, all_rewards):
        avg_reward = sum(all_rewards) / len(all_rewards)
        if avg_reward > self.best_reward:
            best_path = f"{self.config.SAVE_PATH}/best_checkpoint.pt"
            print(f"Saving the model that achived the best rewards so far into {best_path}")
            self.best_reward = avg_reward
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

