import sys
sys.path.insert(0, './src')
from node import Node

import random
import math
from random import choices



class Simulator:
    def __init__(self, num_consumers, num_producers, grid_rows = 3, grid_cols = 3):

        self.NUM_REQUESTS_PER_CONSUMER = 10
        self.NUM_CONTENT_TYPES = 10
        self.CACHE_SIZE = 0.1 * self.NUM_CONTENT_TYPES

        self.consumers = []
        self.producers = []
        self.routers = []
       
        num_routers = grid_rows * grid_cols

        # assign consumers and producers with gateway routers
        for i in range(num_consumers):
            self.consumers.append(
                Node({'' : random.randint(0, num_routers - 1)}, 0)
            )
        for i in range(num_producers):
            self.producers.append(
                Node({'' : random.randint(0, num_routers - 1)}, 0)
            )

        # init grid topology matrix for routers
        adj_mtx = []
        for i in range(num_routers):
            adj_mtx.append([])
            for j in range(num_routers):
                if i == j:
                    adj_mtx[-1].append(1)
                else:
                    adj_mtx[-1].append(0)
        
        # connect routers in a grid
        for i in range(num_routers):
            row_ix = i // grid_cols % grid_rows
            col_ix = i % grid_cols
            for j in range(num_routers):
                pair_row_ix = j // grid_cols % grid_rows
                pair_col_ix = j % grid_cols
                if abs(pair_row_ix - row_ix) + abs(pair_col_ix - col_ix) == 1:
                    adj_mtx[i][j] = 1

        # set fib for routers
        for i in range(num_routers):
            fib = {}
            for p in self.producers:
                if i != p.get_gateway():
                    fib[p.get_name()] = self.get_best_hop(adj_mtx, i, p.get_gateway())
            self.routers.append(
                Node(fib, self.CACHE_SIZE)
            )
    
        #set content names
        content_types = ['a' + str(x) for x in range(0, self.NUM_CONTENT_TYPES)]
        
        #generate probability distribution
        ZIPF_S = 1.2
        zipf_weights = [(1/k**ZIPF_S)/ (sum([1/n**ZIPF_S for n in range(1, self.NUM_CONTENT_TYPES+1)])) for k in range(1,self.NUM_CONTENT_TYPES+1)]

        #make content requests 
        for i in range(0, self.NUM_REQUESTS_PER_CONSUMER):
            for consumer in self.consumers:
                consumer.request(choices(content_types, zipf_weights)[0])


    def get_shortest_path(self, mtx, src, dest):
        visited = set()
        q = [[src]]
        while q:
            path = q.pop(0)
            front = path[-1]
            visited.add(front)
            if front == dest:
                return path
            for neighbor, indicator in enumerate(mtx[front]):
                if indicator == 1 and neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    q.append(new_path)

    def get_best_hop(self, mtx, src, dest):
        return self.get_shortest_path(mtx, src, dest)[1]

if __name__ == "__main__":
    Simulator(2,2)
