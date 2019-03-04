import sys
sys.path.insert(0, './src')
from .src.router import Router
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
        
        self.sp = []
        for i in range(self.num_routers):
            self.sp.append(self.set_shortest_paths(i))

        #Set neighbors 
        for idx, r in enumerate(self.routers):
            neighbors = []
            if idx % grid_cols:
                neighbors.append(self.routers[idx-1])
            if idx % grid_cols < grid_cols -1:
                neighbors.append(self.routers[idx+1])
            if idx > grid_cols:
                neighbors.append(self.routers[idx-grid_cols])
            if idx < self.num_routers - grid_cols:
                neighbors.append(self.routers[idx+grid_cols])
            r.set_neighbors(neighbors)
            print("Router: " + str(idx))
            print([x.name for x in neighbors])

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

    def set_shortest_paths(self, src):
        Q=set()
        dist=[]
        prev=[]
        for r in range(self.num_routers):
            dist.append(float("inf"))
            prev.append(None)
            Q.add(r)
        dist[src] = 0
        prev[src] = src
        while Q:
            u, dist_u = None, float("inf")
            for v in Q:
                if dist[v] < dist_u:
                    u, dist_u = v, dist[v]
            Q.remove(u)
            for neighbor, indicator in enumerate(self.adj_mtx[u]):
                if indicator and dist_u+1 < dist[neighbor]:
                    dist[neighbor]=dist_u+1
                    prev[neighbor]=u
        result = []
        for r in range(self.num_routers):
            cur = r
            while prev[cur] != src:
                cur = prev[cur]
            result.append([cur,dist[r]])
        return result

    def reset_paths(self, producers):
        for src in self.routers:
            paths = self.sp[int(src.name[1:])]
            for p in producers:
                c = p.content
                old=src.FIB[c]
                if src.FIB[c] == p:
                    if src.contentstore.has(c):
                        src.FIB[c] = src
                    return 
                src.FIB[c] = self.routers[paths[int(p.gateway.name[1:])][0]] #reset path to point to gateway
                if old != src.FIB[c]:
                    print("GATE: " +old.name + ", CHOSEN: " + src.FIB[c].name)
                    # print("ERR")
                distance = paths[int(p.gateway.name[1:])][1] + 1
                for dest in self.routers:
                    if dest.contentstore.has(c) and paths[int(dest.name[1:])][1] < distance:
                        src.FIB[c] = self.routers[paths[int(p.gateway.name[1:])][0]]
                        distance = paths[int(dest.name[1:])][1]
