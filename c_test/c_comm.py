import threading
import stat, os
from collections import defaultdict
import numpy as np


num_of_routers = 10
content_types = 200

def compute(num):
    pass

def calc_rankings(history):
    rankings = "meow"
    return rankings


def worker(num):
    path = "/tmp/fifo%d" % num

    if not stat.S_ISFIFO(os.stat(path).st_mode):
        os.mkfifo(path)
    print(path)

    history = defaultdict(int)
    history_np = np.zeros(content_types,int)

    while 1:
        reqs = ""
        fifo = open(path, "r")
        info =  fifo.read().rstrip()
        fifo.close()
        info_split = info.split()
        # print("%s from %d" % (info,num))
        if info_split[0] == "refresh":
            print(history_np)
            rankings=calc_rankings(history)
            history = defaultdict(int)
            history_np = np.zeros(content_types,int)
        elif info_split[0] == "interest":
            history[info_split[1]] += 1
            history_np[int(info_split[1])] +=1
        elif info_split[0] == "data":
            # Send cache of not cache. 
            fifo = open(path,'w')
            fifo.write('OK\n')
            fifo.close()
            
        
        fifo = open(path,'w')
        fifo.write('OK\n')
        fifo.close() 


comm_threads = []
for i in range(num_of_routers):
    t = threading.Thread(target=worker, args=(i,))
    comm_threads.append(t)
    t.start()