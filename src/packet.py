""" A CCN Packet

    Attrributes: 
        name: String to address pkt
        hop_count: no of network hops made by pkts so far
        is_interest: True if pkt is Interest pkt, False if pkt is data pkt
"""
class Packet:
    def __init__(self, name, is_interest = True, hop_count = 0):
        self.name = name
        self.hop_count = hop_count
        self.is_interest = is_interest