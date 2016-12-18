import logging
import random

from numpy import in1d
from numpy.random import exponential
from numpy.random.mtrand import choice
from simpy import Container

from abm.config import TICK_SIZE, DIE_AFTER_N_STARVATION, FINISH_ISLAND, START_ISLAND, EXP_LAMBDA, MIN_POP_BREAK_SIZE, \
    MAX_CHILD_POPULATIONS, LIMIT_SPLITS, MAX_POPULATION_SIZE
from abm.resources import Island

logger = logging.getLogger(__name__)


class Population(object):
    """
    :type env: simpy.Environment
    :type stats: abm.stats.StatTracker
    :type id: str
    :type islands: dict[int, abm.resources.Island]
    :type start_island: int
    :type finish_island: int
    :type current_island: abm.resources.Island
    """

    def __init__(self, env, pop_id, stats, islands, start_island, finish_island, ignore_islands,
                 consumption_rate=1.0, move_propensity=1.0, die_after=100, tick_size=1, ignore_start_islands=None,
                 population_size=None, isolate=True):
        """
        :type env: simpy.Environment
        :type stats: abm.stats.StatTracker
        :type pop_id: int
        :type islands: dict[int, abm.resources.Island]
        :type start_island: int
        """
        self.ignore_start_islands = ignore_start_islands if ignore_start_islands is not None else {start_island}
        self.population_size = population_size
        self.path = [(0, start_island)]
        self.id = pop_id
        self.env = env
        self.stats = stats
        self.backtrack = []

        self.start_island = start_island
        self.finish_island = finish_island

        self.move_propensity = move_propensity
        self.consumption_rate = consumption_rate

        self.visited = {self.start_island} | ignore_islands
        self.islands = islands if not isolate else {
            node: Island(
                env=self.env,
                id=island.id,
                area=island.area,
                perimeter=island.perimeter,
                can_see=island._can_see,
                replenish_rate=island.replenish_rate
            ) for node, island in islands.iteritems()
            }
        self.current_island = islands[self.start_island]
        self.start = self.env.now
        self.tick_size = tick_size

        self.current_request = None

        self.dead = False
        self.die_after = die_after
        self.times_starved = 0

        self._run = self.env.process(self.run())
        self._consume = self.env.process(self.consume())

        if self.current_island.id not in {START_ISLAND, FINISH_ISLAND}:
            self.stats.add_pop(self)

    @property
    def origin(self):
        return self.id.split(':', 1)[0]

    def weighted_choice(self, islands, visited, probs=None):
        """
        :type islands: numpy.ndarray
        :type probs: numpy.ndarray
        :type visited: set[int]
        :rtype: int
        """
        full = {x for x in islands if
                x in self.islands and self.islands[x].r.capacity == self.islands[x].r.count} | visited
        if self.current_island.id == START_ISLAND:
            full |= self.ignore_start_islands
        mask = in1d(islands, list(full), invert=True)
        m = islands[mask]
        p = probs[mask] / probs[mask].sum() if probs is not None else None
        if len(m) > 0:
            c = choice(m, size=1, p=p)
            if len(c) > 0:
                return c[0]

    @property
    def finished(self):
        """
        :rtype: bool
        """
        return self.dead or self.current_island.id == self.finish_island

    @property
    def will_move(self):
        """
        :rtype: bool
        """
        return self.current_island.id == self.start_island or \
               self.current_island.resource_usage < random.expovariate(10.0 / self.move_propensity)

    def _update_island_callback(self, new_island):
        def callback(event):
            """
            :type event: simpy.events.Event
            """
            if event.ok and new_island.id != self.current_island.id:
                self._update_island(new_island)

        return callback

    def _update_island(self, new_island):
        """
        :type new_island: abm.resources.Island
        """
        self.stats.traverse(self.origin, self.current_island.id, new_island.id)
        if self.current_island.id != new_island.id:
            self.path.append((self.env.now, new_island.id))
        self.current_island = new_island

    def _move_to(self, new_island):
        """
        :type new_island: abm.resources.Island
        :rtype: simpy.events.Event
        """
        release_old = self.current_island.r.release(self.current_request)
        current_request = new_island.r.request()
        return release_old & current_request

    def go_back(self):
        """
        :rtype: simpy.events.Event
        """
        if len(self.backtrack) == 0:
            self.backtrack.append(self.start_island)
            logger.info("POPULATION %s IS A FAILURE!!!!", self.id)
            return self.die()
        backtrack_island = self.islands[self.backtrack.pop(-1)]
        logger.debug('%05d Population %s is stuck on %s. Moving back to previous island %s', self.env.now, self.id,
                     self.current_island, backtrack_island)
        e = self._move_to(backtrack_island)
        e.callbacks.append(self._update_island_callback(backtrack_island))
        return e

    def die(self):
        """
        :rtype: simpy.events.Event
        """
        self.dead = True
        self.stats.die(self)
        return self.current_island.r.release(self.current_request)

    def move_island(self):
        """
        :rtype: simpy.events.Event
        """
        mt = self.weighted_choice(self.current_island.can_see, self.visited, self.current_island.probs)
        if mt is None:
            mt = self.weighted_choice(self.current_island.can_see, {self.start_island}, self.current_island.probs)
            # return self.go_back()
        if mt is None:
            return self.go_back()
        move_to = self.islands[mt]
        if move_to.r.capacity > move_to.r.count:
            self.visited.add(move_to.id)
            self.backtrack.append(self.current_island.id)
            logger.debug('%05d Population %s moved from %s to %s', self.env.now, self.id, self.current_island,
                         move_to)
            e = self._move_to(move_to)
            e.callbacks.append(self._update_island_callback(move_to))
            return e
        else:
            logger.info(
                '%05d Population %s tried to move from %s to %s but were rebuffed by existing population',
                self.env.now, self.id, self.current_island, move_to)
            return self.env.event().succeed()

    @property
    def consumption_amount(self):
        """
        :rtype: float
        """
        return self.consumption_rate * self.tick_size

    def unstarve(self):
        if self.times_starved:
            if self.times_starved > 1:
                self.times_starved -= 1
            else:
                self.times_starved = 0

    def starve(self, to_consume):
        self.times_starved += min(1, (self.consumption_amount - to_consume))
        if self.times_starved >= self.die_after:
            logger.info("%05d Population %s STARVED TO DEATH on %s", self.env.now, self.id, self.current_island)
            self.die()
        logger.debug("%05d Population %s is STARVING and consumed %sKM of %s (%0.1f/%0.1f) coastline",
                     self.env.now, self.id, to_consume, self.current_island, self.current_island.resource_level,
                     self.current_island.capacity)

    def consume(self):
        while not self.finished:
            if self.current_island.resource_level >= self.consumption_rate * self.tick_size:
                yield self.current_island.c.get(self.consumption_rate * self.tick_size)
                self.unstarve()
                logger.debug("%05d Population %s consumed %sKM of %s (%0.1f/%0.1f) coastline",
                             self.env.now, self.id, self.consumption_amount, self.current_island,
                             self.current_island.resource_level, self.current_island.capacity)
            else:
                to_consume = self.current_island.resource_level
                if to_consume > 0:
                    yield self.current_island.c.get(to_consume)
                self.starve(to_consume)
            yield self.env.timeout(self.tick_size)

    def run(self):
        yield self.env.timeout(self.tick_size)
        while not self.finished:
            if self.will_move:
                yield self.move_island()
            yield self.env.timeout(self.tick_size)
        if not self.dead:
            self.stats.make_it(self)


class VaryingPopulation(Population):
    def __init__(self, env, pop_id, population_size, stats, islands, start_island, finish_island, ignore_islands,
                 consumption_rate=1.0, move_propensity=1.0, die_after=100, tick_size=1, ignore_start_islands=None,
                 isolate=False):
        # self.demography = zeros((2, 100), dtype=uint32)

        self.child_populations = []
        self.stasis = False
        self.moving = False
        super(VaryingPopulation, self).__init__(env, pop_id, stats, islands, start_island,
                                                finish_island, ignore_islands, consumption_rate, move_propensity,
                                                die_after, tick_size, ignore_start_islands=ignore_start_islands,
                                                population_size=population_size, isolate=isolate)

    def _update_island(self, new_island):
        """
        :type new_island: abm.resources.Island
        """
        # logger.info()
        if self.current_island.id != new_island.id:
            if len(self.path) > 1:
                for i in xrange(len(self.path) - 1):
                    (_, a), (_, b) = self.path[i:i + 2]
                    self.stats.genetic_traverse(self.origin, a, b)
            self.stats.genetic_traverse(self.origin, self.current_island.id, new_island.id)
            self.stats.traverse(self.origin, self.current_island.id, new_island.id)
            logger.debug(
                'Path: %s',
                [
                    (
                        self.path[i:i + 2],
                        self.stats.g[self.path[i][1]][self.path[i + 1][1]][self.origin]['path']
                    ) for i in xrange(len(self.path) - 1)
                    ]
            )
        if not self.moving:
            new_pop = int(random.uniform(0.3, 0.4) * self.population_size)
            stay_pop = self.population_size - new_pop
            new_population = VaryingPopulation(
                env=self.env,
                pop_id='{}:{}'.format(self.id, new_island.id),
                population_size=new_pop,
                stats=self.stats,
                islands=self.islands,
                start_island=new_island.id,
                finish_island=self.finish_island,
                ignore_islands=self.visited,
                consumption_rate=self.consumption_rate,
                move_propensity=self.move_propensity,
                die_after=self.die_after,
                tick_size=self.tick_size,
                ignore_start_islands=self.ignore_start_islands
            )
            if self.current_island.id != new_island.id:
                new_population.path = [x for x in self.path] + [(self.env.now, new_island.id)]
            else:
                new_population.path = [x for x in self.path]

            self.child_populations.append(new_population)
            self.population_size = stay_pop
            new_island.populations.add(new_population)
        else:
            if self in self.current_island.populations:
                self.current_island.populations.remove(self)
            if self.current_island.id != new_island.id:
                self.path.append((self.env.now, new_island.id))
            self.current_island = new_island
            self.current_island.populations.add(self)

    def go_back(self):
        return self.env.event().succeed()

    @property
    def finished(self):
        if LIMIT_SPLITS:
            if len(self.child_populations) >= MAX_CHILD_POPULATIONS:
                self.stasis = True
        return super(VaryingPopulation, self).finished or self.stasis

    def _transfer_request(self, event):
        self.current_request = self._tmp_request
        self._tmp_request = None

    def _move_to(self, new_island):
        """
        :type new_island: abm.resources.Island
        :rtype: simpy.events.Event
        """
        self._tmp_request = new_island.r.request()
        if self.population_size < MIN_POP_BREAK_SIZE:
            self.moving = True
            req = (
                self._tmp_request &
                self.current_island.r.release(self.current_request) &
                self.env.timeout(self.tick_size)
            )
            req.callbacks.append(self._transfer_request)
            return req
        self.moving = False
        return self._tmp_request & self.env.timeout(self.tick_size)

    @property
    def will_move(self):
        logger.debug("%05d Population %s: %s %.2f", self.env.now, self.id, self.current_island,
                     self.current_island.free_carrying_capacity)
        on_start_island = self.current_island.id == START_ISLAND
        near_density = self.current_island.free_carrying_capacity < exponential(0.1)
        return (on_start_island or self.too_populous or near_density) and not self.stasis

    @property
    def too_populous(self):
        return self.population_size > MAX_POPULATION_SIZE

    def move_island(self):
        """
        :rtype: simpy.events.Event
        """
        if self.too_populous \
            and self.current_island.r.capacity > self.current_island.r.count \
            and self.current_island.id != START_ISLAND:
            mt = self.current_island.id
        else:
            mt = self.weighted_choice(self.current_island.can_see, self.visited, self.current_island.probs)
        if mt == FINISH_ISLAND:
            self.visited |= {mt}
        if mt is None:
            # self.child_populations.append(None)
            return self.env.event().succeed()
        else:
            move_to = self.islands[mt]
            if move_to.r.capacity > move_to.r.count:
                logger.debug(
                    '%05d Population %s spawned new population from island %s to %s',
                    self.env.now, self.id, self.current_island, move_to
                )
                e = self._move_to(move_to)
                e.callbacks.append(self._update_island_callback(move_to))
                return e
            else:
                logger.debug(
                    '%05d Population %s tried to move from %s to %s but were rebuffed by existing population',
                    self.env.now, self.id, self.current_island, move_to)
                return self.env.event().succeed()

    def breed(self):
        """
        Survival Ratio:
          0-15:  0.55-0.65
          15-50: 0.41-0.85
        Reproductive Age:
          15-50
        TFR:
          6-8 children per woman per fertility window
        Female Ratio:
          0.5
        """

        p = self.population_size

        # 1/4 of women are of breeding age. 1 in 5 women of breeding age have a baby per year.
        r = float(self.tick_size) * 0.25 * (random.uniform(6.0, 8.0) / 35)
        d = float(self.tick_size) * (0.25 * random.uniform(0.35, 0.45) / 15 + 0.75 * random.uniform(0.15, 0.59) / 35)

        new_pop = int(round(p + p * (r - d) * self.current_island.free_carrying_capacity))
        logger.debug("%05d Population %s: %.2f + %.2f * %.2f * %.2f -> %.2f", self.env.now, self.id,
                     p, p, r, self.current_island.free_carrying_capacity, new_pop)
        self.population_size = new_pop

    def consume(self):
        yield self.env.timeout(self.tick_size)
        while True:
            self.breed()
            yield self.env.timeout(self.tick_size)

    def run(self):
        yield self.env.timeout(self.tick_size)
        while not self.finished:
            if self.will_move:
                self.current_request = yield self.move_island()
            yield self.env.timeout(self.tick_size)
        if not self.dead and not self.stasis:
            self.stats.make_it(self)


def population(env, name, islands, start_island, g, consumption_rate=1, move_propensity=1):
    """

    :param env:
    :type env: Environment
    :param name:
    :type name: int
    :param islands:
    :type islands: dict[int, abm.resources.Island]
    :param start_island:
    :type start_island: int
    :param consumption_rate:
    :type consumption_rate: int
    :param move_propensity:
    :type move_propensity: int
    """

    times_starved = 0
    dead = False
    # route = []
    island_containers = {}
    visited = {1630, 1957, 1766, 1771, 1737, 1802}
    backtrack = []
    global total_dead
    global in_australia

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

    current_island = start_island
    current_request = yield islands[current_island].r.request() | env.timeout(100)
    start = env.now
    i = islands[current_island]
    visited.add(i.id)
    logger.debug("%05d Population %s appears on %s at %s", env.now, name, start_island, start)
    yield env.timeout(TICK_SIZE)

    while i.id != FINISH_ISLAND:
        will_move = i.id == START_ISLAND or (island_containers[i.id].resource_level / island_containers[
            i.id].capacity) < random.expovariate(EXP_LAMBDA) * move_propensity
        if will_move:
            mt = weighted_choice(i.can_see, i.probs, visited)
            if mt is None:
                if len(backtrack) == 0:
                    logger.debug("POPULATION %s IS A FAILURE!!!!", name)
                    dead = True
                    total_dead += 1
                    break
                backtrack_island = islands[backtrack.pop(-1)]
                logger.debug('%05d Population %s is stuck on %s. Moving back to previous island %s', env.now, name, i,
                             backtrack_island)
                release_old = i.r.release(current_request)
                current_request = backtrack_island.r.request()
                yield release_old & current_request
                # route.append((env.now, i.id, backtrack_island.id))
                if 'traversals' not in g.edge[i.id][backtrack_island.id]:
                    g.edge[i.id][backtrack_island.id]['traversals'] = 1
                else:
                    g.edge[i.id][backtrack_island.id]['traversals'] += 1
                i = backtrack_island
            else:
                move_to = islands[mt]
                if move_to.r.capacity > move_to.r.count:
                    release_old = i.r.release(current_request)
                    current_request = move_to.r.request()
                    yield release_old & current_request
                    visited.add(move_to.id)
                    backtrack.append(i.id)
                    # route.append((env.now, i.id, move_to.id))
                    if 'traversals' not in g.edge[i.id][move_to.id]:
                        g.edge[i.id][move_to.id]['traversals'] = 1
                    else:
                        g.edge[i.id][move_to.id]['traversals'] += 1
                    logger.debug('%05d Population %s moved from %s to %s', env.now, name, i, move_to)
                    i = move_to
                else:
                    logger.info(
                        '%05d Population %s tried to move from %s to %s but were rebuffed by existing population',
                        env.now, name, i, move_to)
                    # else:
                    #     logger.debug('{:05d} Population {} is stuck on {} and will probably die'.format(env.now, name, i))
        if i.id not in island_containers:
            island_containers[i.id] = Container(env, capacity=i.perimeter, init=i.perimeter)
        if island_containers[i.id].resource_level >= consumption_rate * TICK_SIZE:
            yield island_containers[i.id].get(consumption_rate * TICK_SIZE)
            if times_starved:
                if times_starved > 1:
                    times_starved -= 1
                else:
                    times_starved = 0

            logger.debug("%05d Population %s consumed %sKM of %s (%0.1f/%0.1f) coastline",
                         env.now, name, consumption_rate * TICK_SIZE, i, island_containers[i.id].resource_level,
                         island_containers[i.id].capacity)
        else:
            to_consume = island_containers[i.id].resource_level
            if to_consume > 0:
                yield island_containers[i.id].get(to_consume)
            times_starved += min(1, (consumption_rate - to_consume))
            if times_starved >= DIE_AFTER_N_STARVATION:
                logger.info("%05d Population %s STARVED TO DEATH on %s", env.now, name, i)
                dead = True
                total_dead += 1
                n = g.node[i.id]
                if 'died' not in n:
                    n['died'] = 1
                else:
                    n['died'] += 1
                yield i.r.release(current_request)
                break
            logger.debug("%05d Population %s is STARVING and consumed %sKM of %s (%0.1f/%0.1f) coastline",
                         env.now, name, to_consume, i, island_containers[i.id].resource_level,
                         island_containers[i.id].capacity)
        yield env.timeout(TICK_SIZE)
    if not dead:
        logger.info("%05d Population %s MADE IT TO AUSTRALIA!!!", env.now, name)
        with open('times.txt', 'a') as nf:
            nf.write('{}\n'.format(env.now - start))
        in_australia += 1
