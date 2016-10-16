import logging
import os
import time
from os.path import join, exists

import networkx as nx
import numpy as np
from simpy.core import Environment, EmptySchedule

from abm.agent.population import VaryingPopulation, Population
from abm.config import START_ISLAND, FINISH_ISLAND, IGNORE_ISLANDS, ENTRY_NORTH, ENTRY_SOUTH, ENTRY_TAIWAN
from abm.resources import Island
from abm.stats import StatTracker
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO
)

NUM_SIMULATIONS = 1
SIM_YEARS = 2000
MAX_FINISHERS = 10
CURRENT_RAINFALL = 2730
RAINFALL_MULTIPLIER = 0.66
RAINFALL = CURRENT_RAINFALL * RAINFALL_MULTIPLIER
RAINFALL_SD = 300
# ENTRY_ISLANDS = ENTRY_TAIWAN
START_POPULATION = 500
ISLAND_SIZE_CUTOFF_KM2 = 10
TICK_SIZE = 1


def sqm_to_deg(sqm):
    return (1.0 / (111.32 ** 2)) * sqm


ISLAND_SIZE_CUTOFF_DEG2 = sqm_to_deg(ISLAND_SIZE_CUTOFF_KM2)

SCENARIOS = (
    # ('all_slow', NUM_SIMULATIONS, SIM_YEARS, VaryingPopulation, (('north', ENTRY_NORTH), ('south', ENTRY_SOUTH), ('taiwan', ENTRY_TAIWAN))),
    # ('north_slow', NUM_SIMULATIONS, SIM_YEARS, VaryingPopulation, (('north', ENTRY_NORTH),)),
    # ('south_slow', NUM_SIMULATIONS, SIM_YEARS, VaryingPopulation, (('south', ENTRY_SOUTH),)),
    # ('taiwan_slow', NUM_SIMULATIONS, SIM_YEARS, VaryingPopulation, (('taiwan', ENTRY_TAIWAN),)),
    ('north_fast', NUM_SIMULATIONS, SIM_YEARS, Population, (('north', ENTRY_NORTH),) * MAX_FINISHERS),
    ('south_fast', NUM_SIMULATIONS, SIM_YEARS, Population, (('south', ENTRY_SOUTH),) * MAX_FINISHERS),
    ('taiwan_fast', NUM_SIMULATIONS, SIM_YEARS, Population, (('taiwan', ENTRY_TAIWAN),) * MAX_FINISHERS),
)

TOTAL = sum(x[1] * x[2] for x in SCENARIOS)

if __name__ == '__main__':
    COUNTER = 0
    SAVE_FOLDER = join('output', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.mkdir(SAVE_FOLDER)
    start_overall = time.time()
    for sim_name, runs, years, pop_cls, entries in SCENARIOS:
        g = nx.read_gpickle(join('data', 'islands.gpickle'))  # type: nx.DiGraph

        num_agents = 1

        save_path = join(SAVE_FOLDER, sim_name)
        if not exists(save_path):
            os.mkdir(save_path)

        for sim_number in xrange(runs):
            logger.info('Running Simulation #%d', sim_number)
            start_sim = time.time()
            # pops = [env.process(population(env, i, islands, START_ISLAND, 1, 1)) for i in xrange(num_agents)]

            env = Environment()
            env.rainfall = max(1, np.random.normal(RAINFALL, RAINFALL_SD))
            islands = {}

            to_del = [node for node, data in g.nodes_iter(data=True) if data['area'] < ISLAND_SIZE_CUTOFF_DEG2]
            g.remove_nodes_from(to_del)

            islands = {node: Island(
                env=env,
                id=node,
                area=data['area'],
                perimeter=data['perimeter'],
                can_see={
                    n: {
                        'ab': d['area'],
                        'ba': g[n][node]['area'],
                        'd': np.sqrt((d['a'][0] - d['b'][0]) ** 2 + (d['a'][1] - d['b'][1]) ** 2)
                    } for n, d in g[node].iteritems()
                    }
            ) for node, data in g.nodes_iter(data=True)}

            stats = StatTracker(g, islands, SIM_YEARS, save_path)  # type: StatTracker
            logger.warning('stats.finished: %s, env.now %s', stats.finished, env.now)
            pops = [
                pop_cls(
                    env=env,
                    pop_id='{}_{:04d}'.format(origin, i),
                    population_size=START_POPULATION,
                    stats=stats,
                    islands=islands,
                    start_island=START_ISLAND,
                    finish_island=FINISH_ISLAND,
                    ignore_islands=IGNORE_ISLANDS,
                    ignore_start_islands=set(g[START_ISLAND].keys()) - entry_locations,
                    tick_size=TICK_SIZE,
                    move_propensity=2,
                    consumption_rate=2
                ) for i, (origin, entry_locations) in enumerate(entries)]
            i = 0
            t = env.now
            while stats.num_in_aus < MAX_FINISHERS and stats.finished < len(pops) and env.now < years:
                if env.now != t:
                    stats.update_populations(env.now)
                    env.rainfall = max(1, np.random.normal(RAINFALL, RAINFALL_SD))
                    t = env.now
                    COUNTER += 1
                    done = (COUNTER / float(TOTAL)) * 100.0
                    # logger.info("%02.4f%% Entry: %s Sim Run: %04d Year: %04d Populations: %04d Total Pop: %04d",
                    #             done, sim_name, sim_number, env.now, stats.num_populations(), stats.total_population())
                try:
                    env.step()
                except EmptySchedule:
                    break
                i += 1
            logger.warning('stats.finished: %s, env.now %s', stats.finished, env.now)
            stats.finish(run_number=sim_number)
            nx.write_gpickle(g, join(save_path, 'traversal_path.gpickle'))
    logger.info("TOTAL TIME TAKEN %0.1f", time.time() - start_overall)
