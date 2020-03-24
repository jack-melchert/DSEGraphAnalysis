from graphviz import Digraph
import sys
import ast
import os

graph = Digraph()

with open(str(sys.argv[1])) as file:
    lines = file.readlines()


with open(".temp/op_types.txt") as file:
    op_types = ast.literal_eval(file.read())

op_types = {str(v): k for k, v in op_types.items()}
graph_num = -1

for line in lines:
    if ':' in line or '#' in line:
        graph_num += 1
    elif 'v' in line:
        graph.node(str(graph_num) + "=" + line.split()[1], op_types[line.split()[2]])
    elif 'e' in line:
        graph.edge(str(graph_num) + "=" + line.split()[1], str(graph_num) + "=" + line.split()[2])
    
if not os.path.exists('pdf'):
    os.makedirs('pdf')

graph.render("pdf/" + str(sys.argv[2]), view=False)