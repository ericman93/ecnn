class History(object):
    def __init__(self):
        self.costs = []

    def add_batch_cost(self, cost):
        self.costs.append(cost)