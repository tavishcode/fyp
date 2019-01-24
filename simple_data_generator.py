import numpy as np
import random
from collections import OrderedDict
import csv
f = open('req_hist_100_million.csv', 'w')
writer = csv.writer(f)

RAND_SEED = 123
NUM_PRODUCERES = 25
ZIPF_S = 1.2
NUM_REQUEST_PER_INTERVAL = 100
NUM_INTERVALS = 1000000 # 1 million rows of data
NUM_REQUESTS = 100000000 # 100 million requests
ZIPF_UPDATE_INTERVAL = 100000 # every 1000 rows of data

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

content_types = ["content" + str(i) for i in range(NUM_PRODUCERES)]
writer.writerow(content_types)
counts = [0 for content in range(NUM_PRODUCERES)]
req_hist = OrderedDict(zip(content_types, counts))

# generate probability distribution
zipf_weights = [(1/k**ZIPF_S)/ (sum([1/n**ZIPF_S for n in range(1, NUM_PRODUCERES+1)])) for k in range(1,NUM_PRODUCERES+1)]
random.shuffle(zipf_weights)

for i in range(1, NUM_REQUESTS+1):
    req_hist[np.random.choice(content_types, 1, p=zipf_weights)[0]] += 1
    if i % NUM_REQUEST_PER_INTERVAL == 0:
        values = []
        for value in req_hist.values():
            values.append(value/NUM_REQUEST_PER_INTERVAL)
        writer.writerow(values)
        req_hist = OrderedDict(zip(content_types, counts))
    if i % ZIPF_UPDATE_INTERVAL == 0:
        random.shuffle(zipf_weights)
        print(i)


