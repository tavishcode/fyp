# Interest packet has only name and hop count. Data packet has name, hop-count and data
class Packet:

    def __init__(self, name, hop_count=0, data=None):
        self.name = name
        self.hop_count = hop_count
        self.data = data

    def is_interest(self):
        if not self.data:
            return True
        return False

    def get_name(self):
        return self.name

    def getSourceName(self):
        return self.name.split('_')[0]