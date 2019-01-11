class Packet:

    def __init__(self, name, is_interest = True):
        self.name = name
        self.hop_count = 0
        self.is_interest = is_interest