import random

class Simulator:

    NUM_REQUESTS_PER_CONSUMER = 10
    NUM_CONTENT_TYPES = 10
    CACHE_SIZE = 0.1 * NUM_CONTENT_TYPES

    consumers = []
    producers = []
    routers = []

    def __init__(self, num_consumers, num_producers, num_routers):
       
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

        # grid topology for routers
        router_mtx = []
        for i in range(num_routers):
            for j in range(num_routers):
                if i == j:
                    router_mtx[i, j] = 1
                else:
                    router_mtx[i, j] = 0

        for i in range(num_routers):
            for j in range(num_routers):         
                if i > 0:
                    router_mtx[i - 1, j] = 1
                if i + 1 < num_routers:
                    router_mtx[i + 1, j] = 1
                if j + 1 < num_routers:
                    router_mtx[i, j + 1] = 1
                if j - 1 > 0:
                    router_mtx[i, j - 1] = 1

        # set fib for routers
        for i in range(num_routers):
            fib = {}
            for p in producers:
                fib.append(p.get_name(), get_best_hop(i, p.get_gateway()))
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
            for neighbor in mtx[front]:
                if neighbor == 1 and neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    q.append(new_path)

    def get_best_hop(mtx, src, dest):
        return get_shortest_path(src, dest)[1]
