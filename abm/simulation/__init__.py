import logging
import os
import time
from os.path import join, exists

import networkx as nx
import numpy as np
from simpy.core import Environment, EmptySchedule

from abm.agent.population import VaryingPopulation
from abm.config import START_ISLAND, FINISH_ISLAND, IGNORE_ISLANDS, ENTRY_NORTH, ENTRY_SOUTH, ENTRY_TAIWAN
from abm.resources import Island
from abm.stats import StatTracker
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NUM_SIMULATIONS = 2
SIM_YEARS = 500
MAX_FINISHERS = 1000
CURRENT_RAINFALL = 2730
RAINFALL_MULTIPLIER = 0.66
RAINFALL = CURRENT_RAINFALL * RAINFALL_MULTIPLIER
RAINFALL_SD = 500
# ENTRY_ISLANDS = ENTRY_TAIWAN
START_POPULATION = 500
ISLAND_SIZE_CUTOFF_KM2 = 10

TOTAL = 3 * NUM_SIMULATIONS * SIM_YEARS


def sqm_to_deg(sqm):
    return (1.0 / (111.32 ** 2)) * sqm


ISLAND_SIZE_CUTOFF_DEG2 = sqm_to_deg(ISLAND_SIZE_CUTOFF_KM2)

if __name__ == '__main__':
    COUNTER = 0
    SAVE_FOLDER = join('output', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.mkdir(SAVE_FOLDER)
    start_overall = time.time()
    for sim_name, entries in (('north', ENTRY_NORTH), ('south', ENTRY_SOUTH), ('taiwan', ENTRY_TAIWAN)):
        g = nx.read_gpickle(join('data', 'islands.gpickle'))  # type: nx.DiGraph

        num_agents = 1

        save_path = join(SAVE_FOLDER, sim_name)
        if not exists(save_path):
            os.mkdir(save_path)

        for sim_number in xrange(NUM_SIMULATIONS):
            start_sim = time.time()
            # pops = [env.process(population(env, i, islands, START_ISLAND, 1, 1)) for i in xrange(num_agents)]

            env = Environment()
            env.rainfall = max(1, np.random.normal(RAINFALL, RAINFALL_SD))
            islands = {}

            to_del = [node for node, data in g.nodes_iter(data=True) if data['area'] < ISLAND_SIZE_CUTOFF_DEG2]
            g.remove_nodes_from(to_del)
            for node, data in g.nodes_iter(data=True):
                islands[node] = Island(
                    env, node, data['area'], data['perimeter'],
                    {n: d['area'] for n, d in g[node].iteritems()}
                )

            stats = StatTracker(g, islands, SIM_YEARS)  # type: StatTracker
            pops = [
                VaryingPopulation(
                    env=env,
                    pop_id=i,
                    population_size=START_POPULATION,
                    stats=stats,
                    islands=islands,
                    start_island=START_ISLAND,
                    finish_island=FINISH_ISLAND,
                    ignore_islands=IGNORE_ISLANDS,
                    ignore_start_islands=set(g[START_ISLAND].keys()) - entries
                ) for i in xrange(num_agents)]
            i = 0
            t = env.now
            while stats.finished < MAX_FINISHERS and env.now < SIM_YEARS:
                if env.now != t:
                    stats.update_populations(env.now)
                    env.rainfall = max(1, np.random.normal(RAINFALL, RAINFALL_SD))
                    t = env.now
                    COUNTER += 1
                    done = (COUNTER / float(TOTAL)) * 100.0
                    logger.info("%02.4f%% Entry: %s Sim Run: %04d Year: %04d", done, sim_name, sim_number, env.now)
                try:
                    env.step()
                except EmptySchedule:
                    break
                i += 1
            stats.finish(save_path=save_path, run_number=sim_number)
            nx.write_gpickle(g, join(save_path, 'traversal_path.gpickle'))
    logger.info("TOTAL TIME TAKEN %0.1f", time.time() - start_overall)
