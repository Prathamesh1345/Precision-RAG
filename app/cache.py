from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from time import monotonic


class LRU:
    """Bounded, expiring cache with defensive copies and a thread-safe API."""

    def __init__(self, max_items=2048, ttl=120):
        self.max_items, self.ttl = max_items, ttl
        self.data = OrderedDict()
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            item = self.data.pop(key, None)
            if item is None or monotonic() - item[0] >= self.ttl:
                return None
            self.data[key] = item
            return deepcopy(item[1])

    def put(self, key, value):
        with self.lock:
            self.data[key] = (monotonic(), deepcopy(value))
            self.data.move_to_end(key)
            while len(self.data) > self.max_items:
                self.data.popitem(last=False)

    def clear(self):
        with self.lock:
            self.data.clear()
