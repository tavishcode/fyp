class Packet:
    def __init__(self, name, is_interest = True, hop_count = 0):
        self.name = name
        self.hop_count = hop_count
        self.is_interest = is_interest