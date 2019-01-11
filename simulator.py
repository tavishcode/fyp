import sys
sys.path.insert(0, './src')
from node import Router, Consumer, Producer
from graph import Graph
from packet import Packet
import random
import math

def visualize(adj_mtx, consumers, producers):
    import matplotlib.pyplot as plt
    import networkx as nx
    gr = nx.Graph()
    for i in range(len(adj_mtx)):
        gr.add_node('r' + str(i))
    for i in range(len(adj_mtx)):
        for j in range(len(adj_mtx[i])):
            if adj_mtx[i][j] == 1:
                gr.add_edge('r' + str(i), 'r' + str(j))
    for c in consumers:
        gr.add_node(c.name)
        gr.add_edge(c.name, c.gateway.name)
    for p in producers:
        gr.add_node(p.name)
        gr.add_edge(p.name, p.gateway.name)
    nx.draw(gr, node_size = 500, with_labels = True)
    plt.show()

class Simulator:

    def __init__(self, num_consumers, num_producers, grid_rows = 3, grid_cols = 3):

        self.NUM_REQUESTS_PER_CONSUMER = 1
        self.NUM_CONTENT_TYPES = num_producers
        self.CACHE_SIZE = 0.1 * self.NUM_CONTENT_TYPES

        self.consumers = []
        self.producers = []
        self.routers = []        
       
        num_routers = grid_rows * grid_cols

        net_core = Graph(grid_rows, grid_cols)

        # assign consumers and producers with gateway routers
        for i in range(num_consumers):
            r = net_core.get_random_router()
            print(r.name)
            self.consumers.append(
                Consumer("c" + str(i), r)
            )

        for i in range(num_producers):
            r = net_core.get_random_router()
            print(r.name)
            self.producers.append(
                Producer("p" + str(i), r, "content" + str(i))
            )


        # set FIBs in routers
        net_core.setRoutesToProducers(self.producers)

        # populate content
        content_types = ["content" + str(i) for i in range(num_producers)]
        
        #generate probability distribution
        ZIPF_S = 1.2
        zipf_weights = [(1/k**ZIPF_S)/ (sum([1/n**ZIPF_S for n in range(1, self.NUM_CONTENT_TYPES+1)])) for k in range(1,self.NUM_CONTENT_TYPES+1)]

        # make content requests
        for i in range(0, self.NUM_REQUESTS_PER_CONSUMER):
            for consumer in self.consumers:
                print("NEW REQUEST")
                pkt = Packet(random.choices(content_types, zipf_weights)[0])
                consumer.request(pkt)
        
        visualize(net_core.adj_mtx, self.consumers, self.producers)


if __name__ == "__main__":
    Simulator(num_consumers = 1, num_producers = 1, grid_rows = 2, grid_cols = 2)
