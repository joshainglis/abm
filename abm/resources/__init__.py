import logging

from numpy import array, int32
from simpy import Resource, Container

from abm.config import DEG_TO_KM, TICK_SIZE

logger = logging.getLogger(__name__)


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
        r_cap = int(area) + 1
        # r_cap = MAX_CAPACITY
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
        self._populations = set()

        self._pop = None
        self._pop_calc_time = None

        self._density = None
        self._density_calc_time = None

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

    @property
    def populations(self):
        """
        :rtype: set[abm.agent.population.Population]
        """
        return self._populations

    @property
    def total_population(self):
        """
        :rtype: int
        """
        if not self.env.now == self._pop_calc_time:
            self._pop = sum(p.population_size for p in self.populations)
            self._pop_calc_time = self.env.now
        return self._pop

    @property
    def carrying_capacity(self):
        """
        :rtype: int
        """

        return self.perimeter * 5

    @property
    def density(self):
        """
        :rtype: float
        """
        if self._density_calc_time != self.env.now:
            logger.debug('island pop: {}, carrying capacity: {}'.format(self.total_population, self.carrying_capacity))
            self._density = 1.0 - min(2.0, self.total_population / float(self.carrying_capacity))
            self._density_calc_time = self.env.now
        return self._density

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
