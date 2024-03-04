from collections import deque

class SingleDataStructure(object):
    def __init__(self, iterable):
        self.ds = deque(iterable)
        self.remove = self.ds.remove
        self.get = self.ds.popleft

    def __get_item__(self, item):
        return self.ds[item]

    def __str__(self):
        return self.ds.__str__()

    def length(self):
        return len(self.ds)

    def __iter__(self):
        return iter(self.ds)


class queue(SingleDataStructure):
    def __init__(self, iterable=[]):
        super().__init__(iterable)
        self.add = self.ds.append