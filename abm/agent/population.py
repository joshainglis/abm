import random
import logging

from numpy import in1d
from numpy.random.mtrand import choice

logger = logging.getLogger(__name__)


class Population(object):
    def __init__(self, env, pop_id, stats, islands, start_island, finish_island, ignore_islands,
                 consumption_rate=1.0, move_propensity=1.0, die_after=100, tick_size=1):
        """
        :type env: Environment
        :type stats: StatTracker
        :type pop_id: int
        :type islands: dict[int, Island]
        :type start_island: int
        """
        self.id = pop_id
        self.env = env
        self.stats = stats
        self.backtrack = []

        self.start_island = start_island
        self.finish_island = finish_island

        self.move_propensity = move_propensity
        self.consumption_rate = consumption_rate

        self.visited = {self.start_island} | ignore_islands
        self.islands = islands
        self.current_island = islands[self.start_island]
        self.start = self.env.now
        self.tick_size = tick_size

        self.current_request = None

        self.dead = False
        self.die_after = die_after
        self.times_starved = 0

    @staticmethod
    def weighted_choice(islands, probs, visited):
        """
        :type islands: numpy.ndarray
        :type probs: numpy.ndarray
        :type visited: set[int]
        :rtype: int
        """
        mask = in1d(islands, list(visited), invert=True)
        m = islands[mask]
        p = probs[mask] / probs[mask].sum()
        if len(m) > 0:
            c = choice(m, size=1, p=p)
            if len(c) > 0:
                return c[0]

    @property
    def finished(self):
        return self.dead or self.current_island.id == self.finish_island

    @property
    def will_move(self):
        return self.current_island.id == self.start_island or \
               self.current_island.resource_usage < random.expovariate(10.0 / self.move_propensity)

    def _move_to(self, new_island):
        release_old = self.current_island.r.release(self.current_request)
        current_request = new_island.r.request()
        yield release_old & current_request
        self.stats.traverse(self.current_island, new_island)
        self.current_island = new_island

    def go_back(self):
        if len(self.backtrack) == 0:
            logger.debug("POPULATION %s IS A FAILURE!!!!", self.id)
            self.die()
            yield None
        else:
            backtrack_island = self.islands[self.backtrack.pop(-1)]
            logger.debug('%05d Population %s is stuck on %s. Moving back to previous island %s', self.env.now, self.id,
                         self.current_island, backtrack_island)
            yield self._move_to(backtrack_island)

    def die(self):
        self.dead = True
        self.stats.die(self)
        yield self.current_island.r.release(self.current_request)

    def move_island(self):
        mt = self.weighted_choice(self.current_island.can_see, self.current_island.probs, self.visited)
        if mt is None:
            yield self.go_back()
        else:
            move_to = self.islands[mt]
            if move_to.r.capacity > move_to.r.count:
                self.visited.add(move_to.id)
                self.backtrack.append(self.current_island.id)
                logger.debug('%05d Population %s moved from %s to %s', self.env.now, self.id, self.current_island,
                             move_to)
                yield self._move_to(move_to)
            else:
                logger.info(
                    '%05d Population %s tried to move from %s to %s but were rebuffed by existing population',
                    self.env.now, self.id, self.current_island, move_to)
                yield None

    @property
    def consumption_amount(self):
        return self.consumption_rate * self.tick_size

    def unstarve(self):
        if self.times_starved:
            if self.times_starved > 1:
                self.times_starved -= 1
            else:
                self.times_starved = 0

    def starve(self):
        to_consume = self.current_island.resource_level
        if to_consume > 0:
            yield self.current_island.c.get(to_consume)
        self.times_starved += min(1, (self.consumption_amount - to_consume))
        if self.times_starved >= self.die_after:
            logger.info("%05d Population %s STARVED TO DEATH on %s", self.env.now, self.id, self.current_island)
            yield self.die()
        logger.debug("%05d Population %s is STARVING and consumed %sKM of %s (%0.1f/%0.1f) coastline",
                     self.env.now, self.id, to_consume, self.current_island, self.current_island.resource_level,
                     self.current_island.capacity)

    def consume(self):
        if self.current_island.resource_level >= self.consumption_rate * self.tick_size:
            yield self.current_island.c.get(self.consumption_rate * self.tick_size)
            self.unstarve()
            logger.debug("%05d Population %s consumed %sKM of %s (%0.1f/%0.1f) coastline",
                         self.env.now, self.id, self.consumption_amount, self.current_island,
                         self.current_island.resource_level, self.current_island.capacity)
        else:
            self.starve()

    def run(self):
        while not self.finished:
            if self.will_move:
                yield self.move_island()
            else:
                yield self.consume()
            yield self.env.timeout(self.tick_size)
        if not self.dead:
            self.stats.make_it(self)
