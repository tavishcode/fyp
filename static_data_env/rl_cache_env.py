import csv
import pandas as pd

class RLCacheEnv:
    def __init__(self, window, cache_size, timesteps, start = None):
        if start == None:
            self.curr_time = window
        else:
            assert(start >= window)
            self.curr_time = start
        self.window = window
        self.cache_size = cache_size
        self.df = pd.read_csv('/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_data_env/req_hist_100_million.csv', sep=',', nrows=timesteps)

    def get_state(self):
        arr = self.df.iloc[self.curr_time - self.window : self.curr_time].values
        return arr

    def get_reward(self, action):
        action = list(enumerate(action))
        action.sort(key = lambda x: x[1], reverse=True)
        # print(action)print
        cache = action[:self.cache_size]
        reward = 0
        for i,a in cache:
            reward += self.df.iat[self.curr_time, i]
        return reward

    def step(self, exploration_action, exploitation_action):
        # print(exploration_action)
        # print(exploitation_action)
        exploration_reward = self.get_reward(exploration_action)
        exploitation_reward = self.get_reward(exploitation_action)
        self.curr_time += 1
        new_state = self.get_state()
        return exploration_reward, exploitation_reward, new_state

