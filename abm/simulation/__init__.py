import logging
from os.path import join

import networkx as nx
from simpy.core import Environment, EmptySchedule

from abm.agent.population import population, Population
from abm.config import START_ISLAND, FINISH_ISLAND, IGNORE_ISLANDS
from abm.resources import Island
from abm.stats import StatTracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
        stats = StatTracker(g)
        pops = [Population(env, i, stats, islands, START_ISLAND, FINISH_ISLAND, IGNORE_ISLANDS) for i in
                xrange(num_agents)]
        while stats.finished < num_agents:
            try:
                env.step()
            except EmptySchedule:
                break
        nx.write_gpickle(g, join('output', 'traversal_path_{}.gpickle'.format(ii)))
