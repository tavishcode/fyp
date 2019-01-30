import sys
sys.path.insert(0, './ddpg_cache')
from ddpg_cache_train import Trainer as ddpg_trainer
from ddpg_cache_buffer import MemoryBuffer
from rl_cache_env import RLCacheEnv
import numpy as np
from matplotlib import pyplot as plt
import torch
import random
import csv

f1 = open('lstm-train-rewards.csv', 'w')
f2 = open('lstm-eval-rewards.csv', 'w')
writer1 = csv.writer(f1)
writer2 = csv.writer(f2)

NUM_CONTENT_TYPES = 25
WINDOW = 3
CACHE_SIZE = 5
PERIOD = 500000
RAND_SEED = 123
assert(PERIOD <= 1000000)
assert(WINDOW <= 1000000)

env = RLCacheEnv(WINDOW, CACHE_SIZE, PERIOD)
ram = MemoryBuffer(1000000)
trainer = ddpg_trainer(NUM_CONTENT_TYPES, NUM_CONTENT_TYPES, ram, window=3, network='lstm')

state = env.get_state()
state = np.float32(state)
state = state.reshape(1, WINDOW, -1)
# print('state', state)
train_rewards = []
eval_rewards = []
num_iters = WINDOW

torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)

while num_iters < PERIOD:
    # print(num_iters)
    exploration_action = trainer.get_exploration_action(state)
    exploitation_action = trainer.get_exploitation_action(state)
    exploration_reward, exploitation_reward, new_state = env.step(exploration_action, exploitation_action)
    train_rewards.append(exploration_reward)
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
        trainer.save_models('lstm')

trainer.save_models('lstm')
writer1.writerow(train_rewards)
writer2.writerow(eval_rewards)
plt.plot(train_rewards)
plt.plot(eval_rewards)
plt.legend(['lstm-train','lstm-eval'])
plt.show()