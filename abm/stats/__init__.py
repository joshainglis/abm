import logging

logger = logging.getLogger(__name__)


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

    @property
    def finished(self):
        return len(self.dead) + len(self.in_aus)

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
