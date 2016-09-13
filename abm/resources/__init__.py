from numpy import array, int32
from simpy import Resource, Container

from abm.config import MAX_CAPACITY, DEG_TO_KM, TICK_SIZE


class Island(object):
    REPR_PARAMS = ['id', 'area', 'perimeter', 'can_see']

    def __init__(self, env, id, area, perimeter, can_see, replenish_rate=0.1):
        """

        :param env:
        :type env: Environment
        :type id: int
        :param area:
        :type area: float
        :param perimeter:
        :type perimeter: float
        :param can_see:
        :type can_see: dict[int, float]
        """
        # r_cap = int(area) + 1
        r_cap = MAX_CAPACITY
        perimeter = int(perimeter * DEG_TO_KM) + 1

        self.env = env
        self.id = id
        self.area = area
        self.perimeter = perimeter
        self.can_see, self.probs = self.normalise(can_see)
        self.replenish_rate = replenish_rate
        self.r = Resource(self.env, capacity=r_cap)
        self.c = Container(self.env, capacity=perimeter, init=perimeter)
        self.action = self.env.process(self.replenish())

    @property
    def resource_level(self):
        return self.c.level

    @property
    def resource_capacity(self):
        return self.c.capacity

    @property
    def resource_usage(self):
        """
        :rtype: float
        """
        return self.resource_level / float(self.resource_capacity)

    @property
    def capacity(self):
        return self.r.capacity

    @property
    def count(self):
        return self.r.count

    @property
    def resources_left(self):
        """
        :rtype: float
        """
        return self.c.level / float(self.c.capacity)

    def normalise(self, can_see):
        """

        :param can_see:
        :type can_see: dict[int, float]
        :return:
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        if len(can_see) > 0:
            islands, probs = zip(*can_see.items())
            probs = array(probs)
            islands = array(islands, dtype=int32)
        else:
            islands, probs = array([]), array([])
        return islands, probs

    def replenish(self):
        while True:
            if self.c.capacity - self.c.level >= self.replenish_rate * TICK_SIZE:
                yield self.c.put(self.replenish_rate * TICK_SIZE)
            yield self.env.timeout(TICK_SIZE)

    def __repr__(self):
        return "Island(%s)", ("%s=%s" % (k, v) for k in self.REPR_PARAMS for v in getattr(self, k))

    def __str__(self):
        return "Island:%s" % (self.id,)
