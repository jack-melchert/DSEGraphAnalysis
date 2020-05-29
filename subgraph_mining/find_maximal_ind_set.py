import networkx as nx
import networkx.algorithms.approximation as nxaa
import regraph as rg
from networkx.drawing.nx_pydot import read_dot
import sys
import matplotlib.pyplot as plt
import pdb
with open(".temp/grami_in.txt") as file:
    lines = file.readlines()

graph = nx.DiGraph()
for line in lines:
    if 'v' in line:
        rg.add_nodes_from(graph, [(line.split()[1], {"type": line.split()[2]})])
    elif 'e' in line:
        rg.add_edges_from(graph, [(line.split()[1], line.split()[2])])


with open("GraMi/Output.txt") as file:
    lines = file.readlines()

graph_ind = -1
graphs = []
for line in lines:
    if ':' in line:
        graph_ind += 1
        graphs.append(nx.DiGraph())
    elif 'v' in line:
        rg.add_nodes_from(graphs[graph_ind], [(line.split()[1], {"type": line.split()[2]})])
    elif 'e' in line:
        rg.add_edges_from(graphs[graph_ind], [(line.split()[1], line.split()[2])])
graphs.reverse()

for idx, pat_graph in enumerate(graphs):
    pattern = pat_graph

    instances = rg.find_matching(graph, pattern)
    # print(instances)


    newNodes = []
    newEdges = []
    for i, inst in enumerate(instances):
        newNodes.append(i)

        for k in inst:
            for z, temp_inst in enumerate(instances):
                for s in temp_inst:
                    if inst[k] is temp_inst[s] and not z == i:
                        if not ((z,i) in newEdges or ((i,z) in newEdges)):
                            newEdges.append((z,i))

        
    # print(newNodes)
    # print(newEdges)

    newGraph = nx.Graph()

    rg.add_nodes_from(newGraph, newNodes)
    rg.add_edges_from(newGraph, newEdges)

    res = nxaa.maximum_independent_set(newGraph)

    # print(res)
    print("PE: ", idx)
    print("Size of PE:" + str(len(pattern.nodes())))
    print("Num subgraphs:" + str(len(newGraph.nodes())))
    print("Num overlapping subgraphs:" + str(len(newGraph.nodes()) - len(res)))
    print("Num PEs used:" + str(len(res)))
    print()


    # pdb.set_trace()

    # color_map = []
    # for node in newGraph:
    #     if node in res:
    #         color_map.append('blue')
    #     else: 
    #         color_map.append('green')      
    # nx.draw_kamada_kawai(newGraph, node_color=color_map, with_labels=True)
    # plt.show()