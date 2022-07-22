import os
import pickle
import subgraph_merging.config as config
from graphviz import Digraph
import networkx as nx
import glob

def convert_coreir_to_dot(directory):

    if not os.path.exists('.temp'):
        os.makedirs('.temp')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    coreir_files = []

    for file in glob.glob(directory + "/*"):
        coreir_files.append(file)

    op_types = {}
    op_types["dummy"] = "0"
    instance_names = {}
    used_ops = set()
    unsupported_ops = set()

    op_index = 1
    out_file = open('.temp/combined_graph.dot', 'w')

    dot = Digraph()
    instance_names["dummy"] = 0
    out_file.write('t # 1\n')
    out_file.write('v ' + str(0) + ' ' + '0' + '\n')
    inst_ind = 1
   
    for ind, f in enumerate(coreir_files):
        
        graph = nx.DiGraph(nx.nx_pydot.read_dot(f))


        for node in graph.nodes(data=True):
            op = node[1]['hwnode'].strip('\"')

            dot.node(node[0] + str(ind), f"{node[0]}\n{op}")

            used_ops.add(op)
            if op not in op_types:
                op_types[op] = op_index
                op_index += 1
            out_file.write('v ' + str(inst_ind) + ' ' + str(op_types[op]) + '\n')
            instance_names[node[0] + str(ind)] = inst_ind
            inst_ind += 1

        source = "dummy"
        sink = node[0] + str(ind)
        dot.edge(source, sink)
        out_file.write('e ' + str(instance_names[source]) + ' ' + str(instance_names[sink]) + ' 0' +'\n')

        for conn in graph.edges:
            source = conn[0] + str(ind)
            sink = conn[1] + str(ind)

            if source in instance_names and sink in instance_names:
                dot.edge(source, sink)
                out_file.write('e ' + str(instance_names[source]) + ' ' + str(instance_names[sink]) + ' 0' +'\n')
                    

    with open('.temp/op_types.txt', 'wb') as op_types_out_file:
        # op_types_out_file.write(str(op_types))
        pickle.dump(op_types, op_types_out_file)

    with open('.temp/used_ops.txt', 'wb') as used_ops_out_file:
        # used_ops_out_file.write(str(used_ops))
        pickle.dump(used_ops, used_ops_out_file)

    dot.render(f'pdf/combined_graph', view=False)  
    print("Used ops:", used_ops)
    out_file.close()

   