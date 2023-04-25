import os.path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BipartiteMatchingGymEnvironment(gym.Env):

    def __init__(self, env_config={}, file_name=''):
        config_defaults = {
            'offline': 100,
            'online': 100,
            'edges': [],
            'time_horizon': 100,
        }

        if not len(file_name):
            file_name = "real_graph/lp_blend/lp_blend.mtx"

        for key, val in config_defaults.items():
            val = env_config.get(key, val)  # Override defaults with constructor parameters
            self.__dict__[key] = val  # Creates variables like self.plot_boxes, self.save_files, etc
            if key not in env_config:
                env_config[key] = val

        print("Start to train on online stochastic matching")

        self.online_type_list = []

        self.read_graph_from_file(file_name)

        # state: offline vertices matched or not, and the adjancent offline vertices
        self.observation_space = spaces.Box(low=np.array([0] * (2 * self.offline) + [0]), high=np.array(
            [1] * (2 * self.offline) + [self.online]), dtype=np.uint32)

        # actions: choose offline vertices to match or not match to any neighbors
        self.action_space = spaces.Discrete(self.offline + 1)

        # set online arrival rate
        self.__set_online_arrival_rate()

    def read_graph_from_file(self, file_name):

        self.edges = []

        with open(file_name, 'r') as rf:
            all_files = rf.readlines()
            m, n = all_files[1].split()[1:]
            m = int(m)
            n = int(n)
            for i in range(n + n):
                self.edges.append([])
            for file in all_files[2:]:
                x, y = file.split()[:2]
                x = int(x) - 1
                y = int(y) - 1
                self.edges[x].append(y + n)
                self.edges[y + n].append(x)

        self.offline = n
        self.online = n
        self.time_horizon = n

    def __set_online_arrival_rate(self):
        if hasattr(self, 'arrival_rate'):
            self.online_arrival_rate = self.arrival_rate
        else:
            self.online_arrival_rate = [1] * self.online

        self.online_arrival_rate = [item / sum(self.online_arrival_rate) for item in self.online_arrival_rate]

    def reset(self):
        self.time_remaining = self.time_horizon

        self.online_type = self.__get_online_type()

        self.online_type_list = []

        self.online_type_list.append(self.online_type)

        self.rl_matching = []

        # an boolean array of offline neighbors that keeps track of unmatched neighbors upon each online vertex arrival
        self.matched_offline_list = [0] * self.offline

        # offline neighbors of online vertex
        adjencent_list = [0] * self.offline
        for y in self.edges[self.online_type]:
            adjencent_list[y - self.online] = 1

        initial_state = self.matched_offline_list + adjencent_list + [self.online_type]

        return initial_state

    def __get_online_type(self):

        online_type = np.random.choice(self.online, p=self.online_arrival_rate)

        return online_type

    def step(self, action):
        done = False
        # set reward of each step
        reward = 0

        if action > self.offline:
            print("Error: offline neighbor do not exist.")
            raise

        elif ((action + self.online) not in self.edges[self.online_type]):
            print("offline is not a valid neighbor")
            self.rl_matching.append(-1)

        elif self.matched_offline_list[action] == 1:
            # can't insert item bin overflow
            print("offline neighbor already matched or choose not to match ")
            self.rl_matching.append(-1)

        elif action == self.offline:
            # choose not to match online vertex
            print("choose not to match ")
            self.rl_matching.append(-1)

        else:  # match offline neighbors
            reward = 1
            self.rl_matching.append(action + self.online)
            self.matched_offline_list[action] = 1

        self.time_remaining -= 1

        if self.time_remaining == 0:
            done = True

        # get the next item
        self.online_type = self.__get_online_type()

        # only add online vertex when not done
        if not done:
            self.online_type_list.append(self.online_type)

            self.realsize = len(self.online_type_list)

        adjencent_list = [0] * self.offline
        for y in self.edges[self.online_type]:
            adjencent_list[y - self.online] = 1

        # state is the whether offline vertices has already matched and adjancent matrix for current arrival online type
        state = self.matched_offline_list + adjencent_list + [self.online_type]

        info = None

        return state, reward, done, info


class BipartiteMatchingActionMaskGymEnvironment(BipartiteMatchingGymEnvironment):
    def __init__(self, env_config={}, file_name=''):
        super().__init__(env_config, file_name)
        self.observation_space = spaces.Dict({
            # a mask of valid actions for adjancent offline neighbors
            "action_mask": spaces.Box(
                0,
                1,
                shape=(self.action_space.n,),
                dtype=np.float32),
            # original observations
            "real_obs": self.observation_space
        })

    def reset(self):
        state = super().reset()

        # only assign online vertex to offline neighbors or do not match
        valid_actions = self.__get_valid_actions()

        self.action_mask = [1 if x in valid_actions else 0 for x in range(self.action_space.n)]

        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }

        return obs

    def step(self, action):
        state, reward, done, info = super().step(action)

        valid_actions = self.__get_valid_actions()

        self.action_mask = [1 if x in valid_actions else 0 for x in range(self.action_space.n)]

        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs, reward, done, info

    def __get_valid_actions(self):
        valid_actions = list()

        # only allow match online vertex to its adjancent unmatched offline neighbors
        for y in self.edges[self.online_type]:
            if not self.matched_offline_list[y - self.online]:
                valid_actions.append(y - self.online)

        # choose not to match the online vertex
        valid_actions.append(self.offline)

        return valid_actions


from algorithms.Max_matching import Max_matching


class StochasticBipartiteMatchingActionMaskGymEnvironment(BipartiteMatchingActionMaskGymEnvironment):
    def optimal_matching_prob(self, sample_num, real_size):
        self.type_prob = list()
        for i in range(self.online):
            self.type_prob.append(dict())
        for count in range(sample_num):
            self.realize(real_size)
            res = Max_matching(self)
            for i in range(real_size):
                online_type = self.online_type_list[i]
                if res[i] != -1:
                    if res[i] in self.type_prob[online_type]:
                        self.type_prob[online_type][res[i]] += 1.0 / sample_num
                    else:
                        self.type_prob[online_type][res[i]] = 1.0 / sample_num

    def realize(self, real_size):
        self.online_type_list = [0] * real_size
        for i in range(real_size):
            self.online_type_list[i] = np.random.randint(0, self.online)
        self.realsize = len(self.online_type_list)

    def get_optimal_matching_prob(self, file_name):

        wf_name = '/'.join(file_name.split("/")[:-1] + ["edge_with_prob.txt"])
        if os.path.exists(wf_name):
            self.type_prob = list()
            for i in range(self.online):
                self.type_prob.append(dict())
            with open(wf_name, 'r') as rf:
                for file in rf.readlines():
                    i,j,p = file.strip().split(",")
                    i = int(i)
                    j = int(j)
                    p = float(p)
                    self.type_prob[i][j] = p
        else:
            self.optimal_matching_prob(10000, self.online)
            with open( wf_name, "w" ) as wf:
                for i in range(self.online):
                    for k in self.type_prob[i].keys():
                        wf.writelines( f"{i}, {k}, {self.type_prob[i][k]}\n")
    def __init__(self, env_config={}, file_name=''):
        super().__init__(env_config, file_name)

        # matching probability of online vertex to offline neighbors
        self.get_optimal_matching_prob(file_name)



        self.observation_space['real_obs'] = spaces.Box(low=np.array([0] * (3 * self.offline) + [0]), high=np.array(
            [1] * (3 * self.offline) + [self.online]), dtype=np.float32)

        # for key in self.type_prob[self.online_type].keys():
        #     online_type_prob [key - self.online] = self.type_prob[self.online_type][key]
    def reset(self):
        obs = super().reset()

        online_type_prob = [0] * self.offline

        for key in self.type_prob[self.online_type].keys():
            online_type_prob [key - self.online] = self.type_prob[self.online_type][key]

        obs['real_obs']  = np.hstack( (obs['real_obs'], online_type_prob))

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        online_type_prob = [0] * self.offline

        for key in self.type_prob[self.online_type].keys():
            online_type_prob [key - self.online] = self.type_prob[self.online_type][key]

        obs['real_obs']  = np.hstack( (obs['real_obs'], online_type_prob))

        return obs, reward, done, info

