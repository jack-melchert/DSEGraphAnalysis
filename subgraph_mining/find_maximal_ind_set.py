import networkx as nx
import networkx.algorithms.approximation as nxaa
import regraph as rg
from networkx.drawing.nx_pydot import read_dot
import sys
import matplotlib.pyplot as plt
import pdb
from networkx.algorithms import isomorphism


def node_match(n1, n2):
    return n1["type"] == n2["type"]


def find_maximal_independent_set(filename, sub_filename):
    with open(filename) as file:
        lines = file.readlines()

    graph = nx.DiGraph()
    for line in lines:
        if 'v' in line:
            rg.add_nodes_from(graph, [(line.split()[1], {"type": line.split()[2]})])
        elif 'e' in line:
            rg.add_edges_from(graph, [(line.split()[1], line.split()[2])])


    with open(sub_filename) as file:
        lines = file.readlines()

    graph_ind = -1
    graphs = []
    invalid_graphs = []
    for line in lines:
        if ':' in line:
            graph_ind += 1
            graphs.append(nx.DiGraph())
        elif 'v' in line:
            rg.add_nodes_from(graphs[graph_ind], [(line.split()[1], {"type": line.split()[2]})])
            # if line.split()[2] == '0':
            #     invalid_graphs.append(graph_ind)
        elif 'e' in line:
            rg.add_edges_from(graphs[graph_ind], [(line.split()[1], line.split()[2])])

    ret = {}

    for idx, pat_graph in enumerate(graphs):
        if idx in invalid_graphs:
            continue
    
        pattern = pat_graph

        test = isomorphism.GraphMatcher(graph, pattern, node_match=node_match)
        instances_gen = test.subgraph_isomorphisms_iter()

        instances = []
        for inst in instances_gen:
            instances.append(inst)

        newNodes = []
        newEdges = []
        for i, inst in enumerate(instances):
            # print(inst)
            newNodes.append(i)

            for k in inst:
                for z, temp_inst in enumerate(instances):
                    for s in temp_inst:
                        # breakpoint()
                        if k == s and not z == i:
                            if not ((z,i) in newEdges or ((i,z) in newEdges)):
                                newEdges.append((z,i))

            
        # print(newNodes)
        # print(newEdges)

        newGraph = nx.Graph()

        rg.add_nodes_from(newGraph, newNodes)
        rg.add_edges_from(newGraph, newEdges)

        res = nxaa.maximum_independent_set(newGraph)

        # print(res)
        print("Subgraph: ", idx)
        print("Size of Subgraph:" + str(len(pattern.nodes())))
        print("Num subgraphs occurences:" + str(len(newGraph.nodes())))
        print("Num overlapping subgraphs:" + str(len(newGraph.nodes()) - len(res)))
        print("Size max ind set:" + str(len(res)))
        print()



        # pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        # nx.draw_networkx(graph, pos, with_labels=True)
        # plt.show()

        # color_map = []
        # for node in newGraph:
        #     if node in res:
        #         color_map.append('blue')
        #     else: 
        #         color_map.append('green')     
        # pos = nx.nx_agraph.graphviz_layout(newGraph, prog='dot')
        # nx.draw_networkx(newGraph, pos, node_color=color_map, with_labels=True) 
        # plt.show()
        ret[idx] = (len(newGraph.nodes()), len(newGraph.nodes()) - len(res), len(res))
    return ret