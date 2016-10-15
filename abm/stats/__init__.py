import logging
from os import mkdir

import numpy as np
from os.path import join, exists

logger = logging.getLogger(__name__)


class StatTracker(object):
    def __init__(self, g, islands, max_years, save_path):
        """
        :type save_path: str
        :type g: networkx.DiGraph
        :type islands: dict[int, abm.resources.Island]
        """
        self.save_path = save_path
        self.g = g
        self.dead = {}
        self.in_aus = {}
        self.islands = islands.values()
        self.pop_history = np.zeros((len(islands), max_years), dtype=np.uint32)
        self._populations = []

        self.logging_path = join(self.save_path, 'logs')
        if not exists(self.logging_path):
            mkdir(self.logging_path)

    def add_pop(self, pop):
        self._populations.append(pop)

    def num_populations(self):
        return len(self._populations)

    def total_population(self):
        return sum(x.population_size for x in self._populations)

    def update_populations(self, time_step):
        """
        :type time_step: int
        """
        self.pop_history[:, time_step] = [island.total_population for island in self.islands]

    @property
    def num_in_aus(self):
        return len(self.in_aus)

    @property
    def finished(self):
        return len(self.dead) + len(self.in_aus)

    def traverse(self, pop_name, from_island, to_island):
        """
        :type pop_name: str
        :type from_island: int
        :type to_island: int
        """
        if pop_name not in self.g.edge[from_island][to_island]:
            self.g.edge[from_island][to_island][pop_name] = {
                'traversals': 1,
                'path': 0,
                'tree': 0,
            }
        else:
            self.g.edge[from_island][to_island][pop_name]['traversals'] += 1

    def path_traverse(self, pop_name, from_island, to_island):
        """
        :type pop_name: str
        :type from_island: int
        :type to_island: int
        """
        if pop_name not in self.g.edge[from_island][to_island]:
            self.g.edge[from_island][to_island][pop_name] = {
                'traversals': 0,
                'path': 1,
                'tree': 0,
            }
        else:
            self.g.edge[from_island][to_island][pop_name]['path'] += 1

    def genetic_traverse(self, pop_name, from_island, to_island):
        """
        :type pop_name: str
        :type from_island: int
        :type to_island: int
        """
        if pop_name not in self.g.edge[from_island][to_island]:
            self.g.edge[from_island][to_island][pop_name] = {
                'traversals': 0,
                'path': 0,
                'tree': 0
            }
        else:
            self.g.edge[from_island][to_island][pop_name]['tree'] += 1

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
        """
        :type pop: abm.agent.population.VaryingPopulation
        :rtype:
        """
        if len(pop.path) > 1:
            for i in xrange(len(pop.path) - 1):
                (_, a), (_, b) = pop.path[i:i + 2]
                self.path_traverse(pop.origin, a, b)

        logger.info("%05d Population %s MADE IT TO AUSTRALIA!!!", pop.env.now, pop.id)
        times = join(self.logging_path, 'times.txt')
        if not exists(times):
            with open(times, 'w') as nf:
                nf.write('population,arrival_time,path\n')
        with open(times, 'a') as nf:
            nf.write(
                '{},{},{}\n'.format(pop.origin, pop.env.now, ':'.join(map(lambda x: '|'.join(map(str, x)), pop.path))))
        self.in_aus[pop.id] = {
            'duration': pop.env.now - pop.start,
            'path': pop.backtrack
        }
        for i in xrange(len(pop.backtrack) - 1):
            a, b = pop.backtrack[i:i + 1]
            self.traverse(pop.origin, a, b)

    def finish(self, run_number=0):
        save_path = join(self.save_path, 'pop_hist')
        if not exists(save_path):
            mkdir(save_path)

        with open(join(save_path, 'population_{:04d}.npy'.format(run_number)), 'wb') as nf:
            np.savez_compressed(nf, history=self.pop_history, map=np.array([i.id for i in self.islands]))
