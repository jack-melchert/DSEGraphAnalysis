import json
import coreir
import sys
import os
import pickle
import subgraph_merging.config as config
from graphviz import Digraph
import networkx as nx
import pydot

def convert_coreir_to_dot(coreir_files):

    if not os.path.exists('.temp'):
        os.makedirs('.temp')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    stripped_files = [os.path.basename(f).replace(".json", "") for f in coreir_files]
    dot_files = [os.path.basename(f).replace(".json", ".dot") for f in coreir_files]

    op_types = {}
    # op_types["self"] = "0"
    instance_names = {}
    used_ops = set()
    unsupported_ops = set()

    op_index = 0

   
    for ind, f in enumerate(coreir_files):
        
        print(f)
        graph = nx.Graph(nx.nx_pydot.read_dot(f))
        out_file = open('.temp/' + dot_files[ind], 'w')

        inst_ind = 0
        # instance_names["self"] = 0
        out_file.write('t # 1\n')
        # out_file.write('v ' + str(0) + ' ' + '0' + '\n')

        dot = Digraph()

        for node in graph.nodes(data=True):

            op = node[1]['hwnode'].strip('\"')

            dot.node(node[0], f"{node[0]}\n{op}")

            used_ops.add(op)
            if op not in op_types:
                op_types[op] = op_index
                op_index += 1
            out_file.write('v ' + str(inst_ind) + ' ' + str(op_types[op]) + '\n')
            instance_names[node[0]] = inst_ind
            inst_ind += 1


        for conn in graph.edges:

            source = conn[0]
            sink = conn[1]

            dot.edge(source, sink)
            if source in instance_names and sink in instance_names:
                out_file.write('e ' + str(instance_names[source]) + ' ' + str(instance_names[sink]) + ' 0' +'\n')
                    

    with open('.temp/op_types.txt', 'wb') as op_types_out_file:
        # op_types_out_file.write(str(op_types))
        pickle.dump(op_types, op_types_out_file)

    with open('.temp/used_ops.txt', 'wb') as used_ops_out_file:
        # used_ops_out_file.write(str(used_ops))
        pickle.dump(used_ops, used_ops_out_file)

    dot.render(f'pdf/{dot_files[ind]}', view=False)  
    print("Used ops:", used_ops)
    out_file.close()

   