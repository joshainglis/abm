import logging
from os import mkdir

import numpy as np
from os.path import join, exists

logger = logging.getLogger(__name__)


class StatTracker(object):
    def __init__(self, g, islands, max_years):
        """
        :type g: networkx.DiGraph
        :type islands: dict[int, abm.resources.Island]
        """
        self.g = g
        self.dead = {}
        self.in_aus = {}
        self.islands = islands.values()
        self.pop_history = np.zeros((len(islands), max_years), dtype=np.uint32)

    def update_populations(self, timestep):
        """
        :type timestep: int
        """
        self.pop_history[:, timestep] = [island.total_population for island in self.islands]

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

    def path_traverse(self, from_island, to_island):
        """
        :type from_island: int
        :type to_island: int
        """
        if 'traversals' not in self.g.edge[from_island][to_island]:
            self.g.edge[from_island][to_island]['path'] = 1
        else:
            self.g.edge[from_island][to_island]['path'] += 1

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

    def finish(self, save_path=None, run_number=0):
        save_path = save_path if save_path is not None else 'output'
        save_path = join(save_path, 'pop_hist')
        if not exists(save_path):
            mkdir(save_path)

        with open(join(save_path, 'population_{:04d}.npy'.format(run_number)), 'wb') as nf:
            np.savez_compressed(nf, history=self.pop_history, map=np.array([i.id for i in self.islands]))
