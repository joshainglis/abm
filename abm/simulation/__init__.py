import logging
import random

import networkx as nx
from numpy import in1d
from numpy.random import choice
from simpy.core import Environment, EmptySchedule
from simpy.resources.container import Container

from abm.resources import Island
from abm.config import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StatTracker(object):
    def __init__(self, g):
        """
        :type g: networkx.DiGraph
        """
        self.g = g
        self.dead = {}
        self.in_aus = {}
        self.paths = {}

    @property
    def num_in_aus(self):
        return len(self.in_aus)

    def traverse(self, from_island, to_island):
        """
        :type from_island: int
        :type to_island: int
        """
        if 'traversals' not in self.g.edge[from_island][to_island]:
            self.g.edge[from_island][to_island]['traversals'] = 1
        else:
            self.g.edge[from_island][to_island]['traversals'] += 1

    def die(self, pop):
        """
        :type pop: abm.agent.population.Population
        """
        self.dead[pop.id] = pop.env.now

        n = self.g.node[pop.current_island.id]
        if 'died' not in n:
            n['died'] = 1
        else:
            n['died'] += 1

    def make_it(self, pop):
        logger.info("%05d Population %s MADE IT TO AUSTRALIA!!!", pop.env.now, pop.id)
        with open('times.txt', 'a') as nf:
            nf.write('{}\n'.format(pop.env.now - pop.start))
        self.in_aus[pop.id] = {
            'duration': pop.env.now - pop.start,
            'path': pop.backtrack
        }
        for i in xrange(len(pop.backtrack) - 1):
            a, b = pop.backtrack[i:i + 1]
            self.traverse(a, b)


def population(env, name, islands, start_island, consumption_rate=1, move_propensity=1):
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


if __name__ == '__main__':
    g = nx.read_gpickle(r'data\islands.gpickle')  # type: nx.DiGraph
    env = Environment()
    islands = {}
    debug = False
    for node, data in g.nodes_iter(data=True):
        islands[node] = Island(
            env, node, data['area'], data['perimeter'],
            {n: d['area'] for n, d in g[node].iteritems()}
        )
    num_agents = 500
    for ii in xrange(20):
        pops = [env.process(population(env, i, islands, START_ISLAND, 1, 1)) for i in xrange(num_agents)]
        total_dead = 0
        in_australia = 0
        while in_australia + total_dead < num_agents:
            try:
                env.step()
            except EmptySchedule:
                break
        nx.write_gpickle(g, 'traversal_path_{}.gpickle'.format(ii))
