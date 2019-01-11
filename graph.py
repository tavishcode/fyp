import sys
sys.path.insert(0, './src')
from node import Router
import random 

class Graph:
    def __init__(self, grid_rows, grid_cols):
        self.CACHE_SIZE = 2

        # init grid topology matrix for routers
        self.num_routers = grid_rows * grid_cols
        self.routers = []

        # init routers
        for i in range(0, self.num_routers):
            name = 'r' + str(i) 
            self.routers.append(Router({}, self.CACHE_SIZE, name)) 

        self.adj_mtx = []

        for i in range(self.num_routers):
            self.adj_mtx.append([])
            for j in range(self.num_routers):
                if i == j:
                    self.adj_mtx[-1].append(1)
                else:
                    self.adj_mtx[-1].append(0)
        
        # connect routers in a grid
        for i in range(self.num_routers):
            row_ix = i // grid_cols % grid_rows
            col_ix = i % grid_cols
            for j in range(self.num_routers):
                pair_row_ix = j // grid_cols % grid_rows
                pair_col_ix = j % grid_cols
                if abs(pair_row_ix - row_ix) + abs(pair_col_ix - col_ix) == 1:
                    self.adj_mtx[i][j] = 1

    def getRouterByName(self, name):
        for r in self.routers:
            if r.name == name:
                return r 
    
    def get_random_router(self):
        return random.choice(self.routers)
    
    def getNeighbors(self):
        pass

    def getNumRouters(self):
        return self.num_routers

    def getRouterNames(self):
        return ['r' + str(i) for i in range(0, self.num_routers)]

    # set fib for routers
    def setRoutesToProducers(self, producers): 
        for r in self.routers:
            fib = {}
            for p in producers:
                if r == p.gateway:
                    fib[p.content] = p
                else:
                    fib[p.content] = self.get_best_hop(self.adj_mtx, self.getPos(r), self.getPos(p.gateway))
            r.FIB = fib

    def getPos(self, router):
        return int(router.name.split('r')[1])

    def getNthRouter(self, n):
        pass

    def get_shortest_path(self, mtx, src, dest):
        # print("DEST: " + str(dest) + " TYPE: " + str(type(dest)))
        visited = set()
        q = [[src]]
        while q:
            path = q.pop(0)
            # print("PATH: " + str(path))
            front = path[-1]
            # print("FRONT: " + str(front))
            visited.add(front)
            if front == dest:
                return path
            for neighbor, indicator in enumerate(mtx[front]):
                if indicator == 1 and neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    q.append(new_path)

    def get_best_hop(self, mtx, src, dest):
        interm = self.get_shortest_path(mtx, src, dest)[1]
        result = self.routers[interm]
        return result
