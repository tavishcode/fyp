import sys
sys.path.insert(0, '../../../ddpg_cache')
sys.path.insert(0, '../../../static_data_env')
from ddpg_cache_train import Trainer as ddpg_trainer
from ddpg_cache_buffer import MemoryBuffer
from rl_cache_env import RLCacheEnv
import numpy as np
from matplotlib import pyplot as plt
import random
import torch
import csv

f1 = open('pretrained-lstm-eval-rewards.csv', 'w')
writer1 = csv.writer(f1)

NUM_CONTENT_TYPES = 25
WINDOW = 3
CACHE_SIZE = 5
PERIOD = 500000
TIMESTEPS = 1000000
RAND_SEED = 123
assert(PERIOD <= 1000000)
assert(WINDOW <= 1000000)

env = RLCacheEnv(WINDOW, CACHE_SIZE, TIMESTEPS, start=500000)
ram = MemoryBuffer(1000000)
trainer = ddpg_trainer(NUM_CONTENT_TYPES, NUM_CONTENT_TYPES, ram, window=3, network='lstm')
trainer.load_models('lstm')

state = env.get_state()
state = np.float32(state)
state = state.reshape(1, WINDOW, -1)
train_rewards = []
eval_rewards = []
num_iters = 1

torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)

while num_iters <= PERIOD:
    exploration_action = trainer.get_exploration_action(state)
    exploitation_action = trainer.get_exploitation_action(state)
    exploration_reward, exploitation_reward, new_state = env.step(exploration_action, exploitation_action)
    # print(exploitation_action)
    eval_rewards.append(exploitation_reward)
    # print(exploration_reward)
    # print(exploitation_reward)
    new_state = np.float32(new_state)
    new_state = new_state.reshape(1, WINDOW, -1)
    ram.add(state, exploration_action, exploration_reward, new_state)
    state = new_state
    trainer.optimize()
    num_iters += 1
    if num_iters % 1000 == 0:
        print('iteration ', num_iters)
    if num_iters % 10000 == 0:
        trainer.save_models('post-pretrained-lstm')

trainer.save_models('post-pretrained-lstm')
writer1.writerow(eval_rewards)
