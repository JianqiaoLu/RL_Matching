import gymnasium as gym
import numpy as np
from gymnasium import spaces




class BipartiteMatchingGymEnvironment(gym.Env):

    def __init__(self, env_config = {}, file_name = ''):
        config_defaults = {
            'offline' : 100,
            'online' : 100,
            'edges':[],
            'time_horizon':1000,
        }

        if not len(file_name):
            file_name = "real_graph/lp_blend/lp_blend.mtx"

        for key, val in config_defaults.items():
            val = env_config.get(key, val)  # Override defaults with constructor parameters
            self.__dict__[key] = val  # Creates variables like self.plot_boxes, self.save_files, etc
            if key not in env_config:
                env_config[key] = val
        print("Start to train on online stochastic matching")

        self.episode_count = 0
        self.online_type_list = []

        self.read_graph_from_file(file_name)

        # state: number of bags at each level, item size,
        self.observation_space = spaces.Box(low=np.array([0] * self.offline + [0]), high=np.array(
            [1] * self.offline + [self.online]), dtype=np.uint32)

        # actions: select a bag from the different levels possible
        self.action_space = spaces.Discrete(self.offline)

        # set online arrival rate
        self.__set_online_arrival_rate()


    def read_graph_from_file(self, file_name):
        self.edges = []
        with open(file_name, 'r') as rf:
            all_files = rf.readlines()
            m,n = all_files[1].split()[1:]
            m  = int(m)
            n = int(n)
            for i in range( n + n ):
                self.edges.append([])
            for file in all_files[2:]:
                x, y = file.split()[:2]
                x = int(x) - 1
                y = int(y) - 1
                self.edges[x].append(y + n)
                self.edges[y + n].append(x)
        self.offline = n
        self.online = n
        self.__set_online_arrival_rate()

    def __set_online_arrival_rate(self):
        if hasattr(self, 'arrival_rate'):
            self.online_arrival_rate =  self.arrival_rate
        else:
            self.online_arrival_rate = [1] * self.online

        self.online_arrival_rate = [item/sum(self.online_arrival_rate) for item in self.online_arrival_rate]

    def reset(self):
        self.time_remaining = self.time_horizon
        self.online_type = self.__get_online_type()
        self.online_type_list = []
        self.online_type_list.append(self.online_type)
        self.num_matched_offline = 0
        self.rl_res = []

        # an boolean array of offline neighbors that keeps track of unmatched neighbors at each level
        self.list_matched_offline = [0] * self.offline

        initial_state = self.list_matched_offline + [self.online_type]

        self.episode_count += 1

        self.step_count = 0
        return initial_state

    def __get_online_type(self):
        online_type = np.random.choice(self.online, p=self.online_arrival_rate)
        return online_type

    def step(self, action):
        done = False
        self.step_count += 1
        if(action == self.offline) :
            import pdb
            pdb.set_trace()

        if action > self.offline:
            print("Error: offline neighbor do not exist.")
            raise

        elif ( (action + self.online ) not in self.edges[self.online_type]):
            print("offline is not a valid neighbor")
            self.rl_res.append(-1)
            reward = 0

        elif self.list_matched_offline[action]  == 1 or action == self.offline:
            # can't insert item because bin overflow
            print("offline neighbor already matched or do not match ")
            self.rl_res.append(-1)
            reward = 0


        else:  # match offline neighbors
            reward = 1
            self.rl_res.append(action + self.online)
            self.list_matched_offline[action] = 1

        self.num_matched_offline += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.online_type = self.__get_online_type()
        if not done:
           self.online_type_list.append(self.online_type)
           self.realsize = len(self.online_type_list)
        # state is the number of bins at each level and the item size
        state = self.list_matched_offline + [self.online_type]

        # info = self.bin_type_distribution_map
        info  = None

        return state, reward, done, info

class BipartiteMatchingActionMaskGymEnvironment(BipartiteMatchingGymEnvironment):
    def __init__(self, env_config={}, file_name = ''):
        super().__init__(env_config, file_name)
        self.observation_space = spaces.Dict({
            # a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
            "action_mask": spaces.Box(
                0,
                1,
                shape=(self.action_space.n,),
                dtype=np.float32),
            "real_obs": self.observation_space
        })

    def reset(self):
        state = super().reset()
        valid_actions = self.__get_valid_actions()
        self.action_mask= [1 if x in valid_actions else 0 for x in range(self.action_space.n)]
        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs

    def step(self, action):
        state, rew, done, info = super().step(action)

        valid_actions = self.__get_valid_actions()
        self.action_mask = [1 if x in valid_actions else 0 for x in range(self.action_space.n)]
        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs, rew, done, info

    def __get_valid_actions(self):
        valid_actions = list()
        #only allow match online vertex to its adjancent unmatched offline neighbors


        for y in self.edges[self.online_type]:
            if not self.list_matched_offline[y - self.online] :
                valid_actions.append(y - self.online)

        return valid_actions



