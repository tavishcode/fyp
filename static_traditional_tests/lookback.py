import sys
sys.path.insert(0,'../static_data_env/')
sys.path.insert(0,'../src/')
from traditional_cache_env import TraditionalCacheEnv
from contentstore import LookbackContentStore
from packet import Packet
import random
import csv

if __name__ == "__main__":

    f = open('lookback-hits.csv', 'w')
    w = csv.writer(f)

    SEED = 123
    TIMESTEPS = 500000

    env = TraditionalCacheEnv(TIMESTEPS, SEED)
    cs = LookbackContentStore(5, 25)
    interval = 0
    hit_ratios = []

    while interval < TIMESTEPS:
        reqs = env.get_requests(interval)
        for req in reqs:
            # print(req)
            found = cs.get(req)
            # print(found)
        # print(cs.timestep_hits)
        hit_ratios.append(cs.timestep_hits/(cs.timestep_hits+cs.timestep_misses))
        cs.update_state()
        interval += 1
        if interval % 10000 == 0:
            print(interval)

    w.writerow(hit_ratios)
