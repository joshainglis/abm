import logging
from os.path import join

import networkx as nx
from simpy.core import Environment, EmptySchedule

from abm.agent.population import VaryingPopulation
from abm.config import START_ISLAND, FINISH_ISLAND, IGNORE_ISLANDS
from abm.resources import Island
from abm.stats import StatTracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SIM_YEARS = 1000

if __name__ == '__main__':
    g = nx.read_gpickle(join('data', 'islands.gpickle'))  # type: nx.DiGraph
    env = Environment()
    islands = {}
    debug = False
    for node, data in g.nodes_iter(data=True):
        islands[node] = Island(
            env, node, data['area'], data['perimeter'],
            {n: d['area'] for n, d in g[node].iteritems()}
        )
    num_agents = 1
    for ii in xrange(1):
        # pops = [env.process(population(env, i, islands, START_ISLAND, 1, 1)) for i in xrange(num_agents)]
        stats = StatTracker(g, islands, SIM_YEARS + 1)
        pops = [VaryingPopulation(env, i, 500, stats, islands, START_ISLAND, FINISH_ISLAND, IGNORE_ISLANDS) for i in
                xrange(num_agents)]
        i = 0
        t = env.now
        while stats.finished < 1000 and env.now < SIM_YEARS:
            if env.now != t:
                stats.update_populations(env.now)
                t = env.now
                logger.info(env.now)
            try:
                env.step()
            except EmptySchedule:
                break
            i += 1
        stats.finish()
        nx.write_gpickle(g, join('output', 'traversal_path_{}.gpickle'.format(ii)))
