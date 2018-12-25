import random
import math

class Simulator:

    NUM_REQUESTS_PER_CONSUMER = 10
    NUM_CONTENT_TYPES = 10
    CACHE_SIZE = 0.1 * NUM_CONTENT_TYPES

    consumers = []
    producers = []
    routers = []

    def __init__(self, num_consumers, num_producers, grid_rows = 3, grid_cols = 3):
       
        num_routers = grid_rows * grid_cols

        # assign consumers and producers with gateway routers
        consumers = []
        for i in range(num_consumers):
            consumers.append(
                Node({'' : random.randint(0, num_routers - 1)}, 0)
            )
        producers = []
        for i in range(num_producers):
            producers.append(
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
            for p in producers:
                if i != p.get_gateway():
                    fib[p.get_name()] = get_best_hop(adj_mtx, i, p.get_gateway())
            routers.append(
                Node(fib, CACHE_SIZE)
            )
    
        # TODO: init content types
        # TODO: init pop rankings for each content type  
        # TODO: set requesting loops for consumers

    def get_shortest_path(mtx, src, dest):
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

    def get_best_hop(mtx, src, dest):
        return get_shortest_path(mtx, src, dest)[1]
