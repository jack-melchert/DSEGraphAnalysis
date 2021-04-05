from graphviz import Digraph
import sys
import ast
import os
import pickle

def graph_output(dot_file, output_filename, max_ind_set_stats = None):

    graph = Digraph()

    with open(dot_file) as file:
        lines = file.readlines()

    with open(".temp/op_types.txt", "rb") as file:
        op_types = pickle.load(file)

    op_types = {str(v): k for k, v in op_types.items()}
    graph_num = -1
    node_count = 0
    for line in lines:
        if ':' in line or '#' in line:
            if graph_num > -1:
                graph.subgraph(subgraph)
            graph_num += 1
            subgraph = Digraph(name="cluster_"+str(graph_num))
            if max_ind_set_stats is not None:
                subgraph.attr(label = f"id:{graph_num}\noccurences:{max_ind_set_stats[graph_num][1][0]}\noverlaps:{max_ind_set_stats[graph_num][1][1]}\nsizeMIS:{max_ind_set_stats[graph_num][1][2]}")
        elif 'v' in line:
            node_count += 1
            subgraph.node(str(graph_num) + "=" + line.split()[1], op_types[line.split()[2]])
        elif 'e' in line:
            # subgraph.edge(str(graph_num) + "=" + line.split()[1], str(graph_num) + "=" + line.split()[2], label=line.split()[3])
            subgraph.edge(str(graph_num) + "=" + line.split()[1], str(graph_num) + "=" + line.split()[2])

    if graph_num == 0:
        graph = subgraph 
        print("num nodes: ", node_count)
    else:
        graph.subgraph(subgraph)   

    if not os.path.exists('pdf'):
        os.makedirs('pdf')
    graph.render("pdf/" + output_filename, view=False)
