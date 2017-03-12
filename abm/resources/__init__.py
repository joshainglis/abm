import logging

from numpy import array, int32
from simpy import Container, Resource

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
        r_cap = int(area / 0.17384) + 1
        # r_cap = MAX_CAPACITY
        perimeter = int(perimeter * DEG_TO_KM) + 1

        self.env = env
        self.id = id
        self.area = area
        self.perimeter = perimeter
        self._can_see = can_see
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

        self._capacity = None
        self._capacity_calc_time = None

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
            if self._pop > 1000 or self.capacity > 1000:
                logger.debug("Island: {}, CarryingCap: {}, Pop: {}".format(self.id, self.capacity, self._pop))
        return self._pop

    @staticmethod
    def birdsell_carrying_capacity(rainfall):
        """

        :param rainfall: yearly rainfall in inches
        :type rainfall: float
        :return: hundreds of square miles that can support 500 people
        :rtype: float
        """
        return 7112.8 * (rainfall ** (-1.58451))

    @staticmethod
    def hundreds_of_square_miles_to_square_kilometers(hsm):
        """

        :param hsm: hundreds of square miles
        :type hsm: float
        :return: square kilometers
        :rtype: float
        """
        return hsm * 258.999

    @classmethod
    def hsm_per_500_person_to_people_per_square_kilometer(cls, hsm_500p):
        """

        :param hsm_500p: hundreds of square miles per 500 person
        :type hsm_500p: float
        :return: People per square kilometer
        :rtype: float
        """
        sk_500p = cls.hundreds_of_square_miles_to_square_kilometers(hsm_500p)
        sk_pp = sk_500p / 500.0
        return sk_pp ** -1

    @staticmethod
    def mm_to_inches(mm):
        """
        :param mm: Millimetres
        :type mm: float
        :return: Inches
        :rtype: float
        """
        return 0.0393701 * mm

    @staticmethod
    def deg_to_sqkm(deg):
        """
        :param deg: Square degrees
        :type deg: float
        :return: square kilometres
        :rtype: float
        """
        return (111.32 ** 2) * deg

    @property
    def carrying_capacity(self):
        """
        :rtype: int
        """
        if self._capacity_calc_time != self.env.now:
            rainfall_inches = self.mm_to_inches(self.env.rainfall)
            hsm_500p = self.birdsell_carrying_capacity(rainfall_inches)
            pp_sk = self.hsm_per_500_person_to_people_per_square_kilometer(hsm_500p)
            self._capacity = pp_sk * self.deg_to_sqkm(self.area)  # * min(5.58, max(1.53, normal(3.348, 1.342)))
            # logger.debug("rainfall_mm: {}, rainfall_inches: {}, ")
        return self._capacity

    @property
    def free_carrying_capacity(self):
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
            probs = [(x['ab'] / self.area) * (x['ba'] / x['d']) for x in probs]
            # logger.info(probs)
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
