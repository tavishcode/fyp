# TODO: add LRU bootstrap for caches that needs to collect initial stats
from collections import OrderedDict
from collections import defaultdict
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


class ContentStore:
  def __init__(self, size):
    self.size = size
    self.hits = 0
    self.misses = 0

  def add(self, item):
    """Decides whether to add item to store and what to evict if store is full"""
    raise NotImplementedError(
        'Base Class ContentStore does not implement a cache')

  def get_helper(self, item):
    """Returns item with item_name if it is in store, else returns None"""
    raise NotImplementedError(
        'Base Class ContentStore does not implement a cache')

  def get(self, item):
    """Wrapper function for get_helper which includes hit/miss statistic updates"""
    item = self.get_helper(item)
    if item != None:
      self.hits += 1
    else:
      self.misses += 1
    return item

  def has(self, item):
    if item.name in self.store:
      return self.store[item.name]
    else:
      return None

  def refresh(self):
    pass


class LruContentStore(ContentStore):
  def __init__(self, size):
    super().__init__(size)
    self.store = OrderedDict()

  def add(self, item):
    if self.size:
      if(len(self.store) == self.size):
        self.store.popitem(last=False)
      self.store[item.name] = item

  def get_helper(self, item):
    try:
      cached_item = self.store.pop(item.name)
      self.store[item.name] = cached_item
      return cached_item
    except:
      return None


class LfuContentStore(ContentStore):
  def __init__(self, size):
    super().__init__(size)
    self.store = {}  # {'name', [item, freq]}

  def add(self, item):
    if self.size:
      if len(self.store) == self.size:
        min_key = None
        min_freq = None
        for key in self.store.keys():
          if min_freq == None or self.store[key][1] < min_freq:
            min_freq = self.store[key][1]
            min_key = key
        self.store.pop(min_key)
      self.store[item.name] = [item, 1]

  def get_helper(self, item):
    try:
      cached_item = self.store[item.name][0]
      self.store[item.name][1] += 1
      return cached_item
    except:
      return None


class RandomContentStore(ContentStore):
  def __init__(self, size):
    super().__init__(size)
    self.rng = np.random.RandomState(123)
    self.store = {}

  def add(self, item):
    if self.size:
      if len(self.store) == self.size:
        self.store.pop(self.rng.choice(list(self.store.keys())))
      self.store[item.name] = item

  def get_helper(self, item):
    try:
      cached_item = self.store[item.name]
      return cached_item
    except:
      return None


class RMAContentStore(ContentStore):
  def __init__(self, size):
    super().__init__(size)
    self.bootstrap = LruContentStore(size)
    self.bootstrap_period = 7
    self.store = {}
    self.history = {}
    self.ranking = defaultdict(int)
    self.interval_count = 0
    self.window = 7
    self.bootstrapping = True

  def add(self, item):
    if self.bootstrapping:
      self.bootstrap.add(item)
    else:
      if self.size:
        if len(self.store) == self.size:
          min_key, min_rank = self.get_min()
          if min_rank != None and min_rank < self.ranking[item.name]:
            self.store.pop(min_key)
            self.store[item.name] = item
        else:
          self.store[item.name] = item

  def get_min(self):
    min_key = None
    min_rank = None
    for key in self.store.keys():
      if min_key == None or self.ranking[key] < min_rank:
        min_rank = self.ranking[key]
        min_key = key
    return min_key, min_rank

  def refresh(self):
    self.interval_count += 1
    if not self.bootstrapping:
      for key in self.ranking.keys():
        if key not in self.history:
          self.history[key] = [0 for _ in range(self.window)]
        self.ranking[key] = sum(self.history[key]) / self.window
    if self.interval_count == self.bootstrap_period:
      self.bootstrapping = False

  def get_helper(self, item):
    try:
      if item.name not in self.history:
        self.history[item.name] = [0 for _ in range(self.window)]
      self.history[item.name][self.interval_count % 7] += 1
      cached_item = self.bootstrap.get(
          item) if self.bootstrapping else self.store[item.name]
      return cached_item
    except:
      return None


class EMAContentStore(ContentStore):
  def __init__(self, size):
    super().__init__(size)
    self.bootstrap = LruContentStore(size)
    self.bootstrap_period = 1
    self.bootstrapping = True
    self.store = {}
    self.interval_count = 0
    self.history = defaultdict(int)
    self.ranking = defaultdict(int)
    self.alpha = 0.1

  def add(self, item):
    if self.bootstrapping:
      self.bootstrap.add(item)
    else:
      if self.size:
        if len(self.store) == self.size:
          min_key, min_rank = self.get_min()
          if min_rank != None and min_rank < self.ranking[item.name]:
            self.store.pop(min_key)
            self.store[item.name] = item
        else:
          self.store[item.name] = item

  def get_min(self):
    min_key = None
    min_rank = None
    for key in self.store.keys():
      if min_key == None or self.ranking[key] < min_rank:
        min_rank = self.ranking[key]
        min_key = key
    return min_key, min_rank

  def refresh(self):
    self.interval_count += 1
    if self.bootstrapping:
      self.ranking = self.history.copy()
    else:
      for key in self.ranking.keys():
        self.ranking[key] = self.ranking[key] + \
            self.alpha*(self.history[key]-self.ranking[key])
    self.history = defaultdict(int)
    if self.interval_count == self.bootstrap_period:
      self.bootstrapping = False

  def get_helper(self, item):
    try:
      self.history[item.name] += 1
      cached_item = self.bootstrap.get(
          item) if self.bootstrapping else self.store[item.name]
      return cached_item
    except:
      return None


class ODContentStore(ContentStore):
  def __init__(self, size):
    super().__init__(size)
    self.bootstrap = LruContentStore(size)
    self.bootstrap_period = 1
    self.bootstrapping = True
    self.interval_count = 0
    self.store = {}
    self.history = defaultdict(int)
    self.ranking = {}

  def add(self, item):
    if self.bootstrapping:
      self.bootstrap.add(item)
    else:
      if self.size:
        if len(self.store) == self.size:
          min_key, min_rank = self.get_min()
          if min_rank != None and min_rank < self.ranking[item.name]:
            self.store.pop(min_key)
            self.store[item.name] = item
        else:
          self.store[item.name] = item

  def get_min(self):
    min_key = None
    min_rank = None
    for key in self.store.keys():
      if min_key == None or self.ranking[key] < min_rank:
        min_rank = self.ranking[key]
        min_key = key
    return min_key, min_rank

  def refresh(self):
    self.interval_count += 1
    self.ranking = self.history.copy()
    self.history = defaultdict(int)
    if self.interval_count == self.bootstrap_period:
      self.bootstrapping = False

  def get_helper(self, item):
    try:
      self.history[item.name] += 1
      cached_item = self.bootstrap.get(
          item) if self.bootstrapping else self.store[item.name]
      return cached_item
    except:
      return None

class PretrainedRNNContentStore(ContentStore):
  def __init__(self, size, online=False):
    super().__init__(size)
    self.bootstrap = LruContentStore(size)
    self.bootstrap_period = 7
    self.bootstrapping = True
    self.window = 7
    self.num_features = 1
    self.model = load_model(
        'drive/My Drive/simple_gru.h5')
    self.store = {}
    self.history = {}
    self.ranking = defaultdict(int)
    self.interval_count = 0
    self.window = 7
    self.scaler = MinMaxScaler()
    self.online = online
    self.train_x = None
    self.train_y = None

  def add(self, item):
    if self.bootstrapping:
      self.bootstrap.add(item)
    else:
      if self.size:
        if len(self.store) == self.size:
          min_key, min_rank = self.get_min()
          if min_rank != None and min_rank < self.ranking[item.name]:
            self.store.pop(min_key)
            self.store[item.name] = item
        else:
          self.store[item.name] = item

  def get_min(self):
    min_key = None
    min_rank = None
    for key in self.store.keys():
      if min_key == None or self.ranking[key] < min_rank:
        min_rank = self.ranking[key]
        min_key = key
    return min_key, min_rank

  def refresh(self):
    self.interval_count += 1
    if not self.bootstrapping:
      agg_data = np.zeros((len(self.history.keys()), self.window))
      for key in self.history.keys():
        agg_data[int(key[7:])] = self.history[key]
      agg_data = self.scaler.fit_transform(agg_data)
      rankings = self.model.predict(agg_data.reshape(-1, 7, 1)).ravel()
      if self.online: # continue to train model during test phase
        self.train_y = agg_data[:,-1]
        if self.train_x is not None: # skips first day
          self.model.train_on_batch(self.train_x, self.train_y)
        self.train_x = agg_data.reshape(-1, 7, 1)
      for i, r in enumerate(rankings):
        self.ranking['content' + str(i)] = r
    if self.interval_count == self.bootstrap_period:
      self.bootstrapping = False

  def get_helper(self, item):
    try:
      if item.name not in self.history:
        self.history[item.name] = np.zeros(self.window)
      self.history[item.name][self.interval_count % 7] += 1
      cached_item = self.bootstrap.get(
          item) if self.bootstrapping else self.store[item.name]
      return cached_item
    except:
      return None
