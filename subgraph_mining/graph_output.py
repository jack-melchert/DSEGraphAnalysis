from graphviz import Digraph
import sys
import ast
import os
import pickle

def graph_output(dot_file, output_filename):

    graph = Digraph()

    with open(dot_file) as file:
        lines = file.readlines()

    with open(".temp/op_types.txt", "rb") as file:
        op_types = pickle.load(file)

    op_types = {str(v): k for k, v in op_types.items()}
    graph_num = -1

    for line in lines:
        if ':' in line or '#' in line:
            if graph_num > -1:
                graph.subgraph(subgraph)
            graph_num += 1
            subgraph = Digraph(name="cluster_"+str(graph_num))
            subgraph.attr(label = str(graph_num))
        elif 'v' in line:
            subgraph.node(str(graph_num) + "=" + line.split()[1], op_types[line.split()[2]])
        elif 'e' in line:
            # subgraph.edge(str(graph_num) + "=" + line.split()[1], str(graph_num) + "=" + line.split()[2], label=line.split()[3])
            subgraph.edge(str(graph_num) + "=" + line.split()[1], str(graph_num) + "=" + line.split()[2])

    graph.subgraph(subgraph)

    if not os.path.exists('pdf'):
        os.makedirs('pdf')

    graph.render("pdf/" + output_filename, view=False)