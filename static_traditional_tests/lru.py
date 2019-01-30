import sys
sys.path.insert(0,'../static_data_env/')
sys.path.insert(0,'../src/')
from traditional_cache_env import TraditionalCacheEnv
from contentstore import LruContentStore
from packet import Packet
import random
import csv

f = open('lru-hits.csv', 'w')
w = csv.writer(f)

SEED = 123
TIMESTEPS = 500000
ZIPF_UPDATE_INTERVAL = 1000 # 1000 rows

env = TraditionalCacheEnv(TIMESTEPS, SEED)
cs = LruContentStore(5)
interval = 0
hit_ratios = []

while interval < TIMESTEPS:
    reqs = env.get_requests(interval)
    for req in reqs:
        found = cs.get(req)
        if found == None:
            cs.add(Packet(req, is_interest=False))
    interval += 1
    if interval % ZIPF_UPDATE_INTERVAL == 0:
        print(interval)
        hit_ratios.append(cs.hits/(cs.hits + cs.misses))
        cs.hits = 0
        cs.misses = 0

w.writerow(hit_ratios)
