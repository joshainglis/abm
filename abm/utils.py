from os.path import join, expanduser

import arcpy
import networkx as nx

import numpy as np

ISLAND_SIZE_CUTOFF_KM2 = 10
TICK_SIZE = 1
OFFSET = 0.01


def sqm_to_deg(sqm):
    return (1.0 / (111.32 ** 2)) * sqm


ISLAND_SIZE_CUTOFF_DEG2 = sqm_to_deg(ISLAND_SIZE_CUTOFF_KM2)

g = nx.read_gpickle(join('data', 'islands.gpickle'))  # type: nx.DiGraph

to_del = [node for node, data in g.nodes_iter(data=True) if data['area'] < ISLAND_SIZE_CUTOFF_DEG2]
g.remove_nodes_from(to_del)

# islands = {
#     node: {
#         other: {
#             'ab': d['area'],
#             'ba': g[other][node]['area'],
#             'd': np.sqrt((d['a'][0] - d['b'][0]) ** 2 + (d['a'][1] - d['b'][1]) ** 2)
#         } for other, d in g[node].iteritems()
#         }
#     for node, data in g.nodes_iter(data=True)
#     }

islands = {}
for node, data in g.nodes_iter(data=True):
    islands[node] = {}
    islands[node]['area'] = data['area']
    islands[node]['perimeter'] = data['perimeter']
    islands[node]['edges'] = {}
    for other, d in g[node].iteritems():
        islands[node]['edges'][other] = {
            'ab': d['area'],
            'ba': g[other][node]['area'],
            'd': np.sqrt((d['a'][0] - d['b'][0]) ** 2 + (d['a'][1] - d['b'][1]) ** 2)
        }

for node, data in islands.iteritems():
    for other, d in data['edges'].iteritems():
        islands[node]['edges'][other]['weight'] = (d['ab'] / islands[node]['area']) * (d['ba'] / d['d'])

for node, data in islands.iteritems():
    for other, d in data['edges'].iteritems():
        g[node][other]['weight'] = islands[node]['edges'][other]['weight']
        g[node][other]['area_ab'] = islands[node]['edges'][other]['ab']
        g[node][other]['area_ba'] = islands[node]['edges'][other]['ba']
        g[node][other]['distance'] = islands[node]['edges'][other]['d']

workspace = join(expanduser('~'), 'Documents', 'Wallacea Viewshed', 'Scratch', 'SL_Analysis', 'sl_-85')
arcpy.env.overwriteOutput = True
spatial_reference = arcpy.Describe(join(workspace, 'gridded_viewpoints.shp')).spatialReference

# RUN = "20161015_200927"

OUTPUT_FOLDER = join(expanduser('~'), 'PycharmProjects', 'abm', 'output')

to_extract = [
    {
        'name': '20161016_123443',
        'scenarios': [
            {'name': 'north_fast', 'folder': 'north_fast', 'origins': ['north']},
            {'name': 'south_fast', 'folder': 'south_fast', 'origins': ['south']},
            {'name': 'taiwan_fast', 'folder': 'taiwan_fast', 'origins': ['taiwan']},
        ]
    }
]

point = arcpy.Point()
array = arcpy.Array()

network = arcpy.CreateFeatureclass_management(
    workspace, "con_weights_east.shp", "POLYLINE", spatial_reference=spatial_reference
)
arcpy.AddField_management(network, 'island_A', field_type='LONG')
arcpy.AddField_management(network, 'island_B', field_type='LONG')
arcpy.AddField_management(network, 'A_sees_B', field_type='DOUBLE')
arcpy.AddField_management(network, 'B_sees_A', field_type='DOUBLE')
arcpy.AddField_management(network, 'distance', field_type='DOUBLE')
arcpy.AddField_management(network, 'weight', field_type='DOUBLE')

cursor = arcpy.da.InsertCursor(network,
                               ["SHAPE@", "island_A", "island_B", "A_sees_B", "B_sees_A", "distance", 'weight'])

for (island_a_id, island_b_id, data) in g.edges_iter(data=True):
    L = data['distance']
    x1 = data['a'][0]
    x2 = data['b'][0]
    if x1 < x2:
        y1 = data['a'][1]
        y2 = data['b'][1]
        x1p = x1 + OFFSET * (y2 - y1) / L
        x2p = x2 + OFFSET * (y2 - y1) / L
        y1p = y1 + OFFSET * (x1 - x2) / L
        y2p = y2 + OFFSET * (x1 - x2) / L
        point.X = x1p
        point.Y = y1p
        array.add(point)
        point.X = x2p
        point.Y = y2p
        array.add(point)

        polyline = arcpy.Polyline(array)
        array.removeAll()
        cursor.insertRow(
            (
                polyline,
                island_a_id,
                island_b_id,
                data['area_ab'],
                data['area_ba'],
                data['distance'],
                data['weight'],
            )
        )
del cursor
