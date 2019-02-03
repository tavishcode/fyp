import sys
sys.path.insert(0, './src')
from router import Router
import random 

"""
    Container class for router nodes
    Generates a Grid-Based network topology and sets routes for nodes
    
    Args: 
        cache_size: capacity of a router node cache
        grid_rows: no of rows in router grid  
        grid_cols: no of columns in router grid
    Attributes:
        routers: list of all router objects 
        num_routers: no of routers in routers
        adj_mtx: adjacency matrix that models connections between router nodes
        
"""
class Graph:
    def __init__(self, cache_size, num_content_types, grid_rows, grid_cols, policy, q, rng):

        """Creates routers and sets node links in adj_mtx"""
        self.routers = []
        self.num_routers = grid_rows * grid_cols
        self.rng = rng
        
        # init routers
        for i in range(0, self.num_routers):
            name = 'r' + str(i)
            self.routers.append(Router(cache_size, num_content_types, name, policy, q)) 

        # set adjacency matrix 
        self.adj_mtx = []

        for i in range(self.num_routers):
            self.adj_mtx.append([])
            for j in range(self.num_routers):
                if i == j:
                    self.adj_mtx[-1].append(1)
                else:
                    self.adj_mtx[-1].append(0)

        for i in range(self.num_routers):
            row_ix = i // grid_cols % grid_rows
            col_ix = i % grid_cols
            for j in range(self.num_routers):
                pair_row_ix = j // grid_cols % grid_rows
                pair_col_ix = j % grid_cols
                if abs(pair_row_ix - row_ix) + abs(pair_col_ix - col_ix) == 1:
                    self.adj_mtx[i][j] = 1
    
    def get_random_router(self):
        return self.rng.choice(self.routers)

    def get_pos(self, router):
        """Returns position of router in grid 
        where pos / grid_rows = row and pos % grid_cols = col"""
        return int(router.name.split('r')[1])

    def get_shortest_path(self, mtx, src, dest):
        """Returns list of node pos for shortest path from src to dest"""
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
        """Returns next router in shortest path from src to dest"""
        interm = self.get_shortest_path(mtx, src, dest)[1]
        result = self.routers[interm]
        return result

    def set_routes_to_producers(self, producers):
        """Sets FIB for shortest path to producers for all routers"""
        for r in self.routers:
            fib = {}
            for p in producers:
                if r == p.gateway:
                    fib[p.content] = p
                else:
                    fib[p.content] = self.get_best_hop(self.adj_mtx, self.get_pos(r), self.get_pos(p.gateway))
            r.FIB = fib



