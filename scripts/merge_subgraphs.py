import networkx as nx
from itertools import count, combinations
import matplotlib.pyplot as plt
from  networkx.drawing import nx_agraph
from networkx.algorithms import bipartite
import pulp
import sys
import pdb
import ast
import math
import os
import json

def ConstructCompatibilityGraph(g1, g2):

    gb = nx.Graph()

    g1_map = {}
    idx = 0
    for n1 in g1.nodes():
        gb.add_node('a' + str(idx), type='node', op=g1.node[n1]['op'], bipartite=0)
        g1_map[n1] = 'a' + str(idx)
        idx += 1

    for (u, v, p) in g1.edges.data('port'):
        gb.add_node(g1_map[u] + ', ' + g1_map[v], type='edge', node0=g1_map[u], node1=g1_map[v], op0=g1.node[u]['op'], op1=g1.node[v]['op'], port=p, bipartite=0)

    g2_map = {}
    idx = 0
    for n1 in g2.nodes():
        gb.add_node('b' + str(idx), type='node', op=g2.node[n1]['op'], bipartite=1)
        g2_map[n1] = 'b' + str(idx)
        idx += 1

    for (u, v, p) in g2.edges.data('port'):
        gb.add_node(g2_map[u] + ', ' + g2_map[v], type='edge', node0=g2_map[u], node1=g2_map[v], op0=g2.node[u]['op'], op1=g2.node[v]['op'], port=p, bipartite=1)


    

    comm_ops =  ["and", "or", "xor", "add", "eq", "mul"]


    left_nodes = [(n,d) for n, d in gb.nodes(data=True) if d['bipartite']==0]
    right_nodes = [(n,d) for n, d in gb.nodes(data=True) if d['bipartite']==1]


    for n0, d0 in left_nodes:
        for n1, d1 in right_nodes:
            if d0['type'] == 'node' and d1['type'] == 'node':
                if d0['op'] == d1['op']:
                    gb.add_edge(n0, n1)
            elif d0['type'] == 'edge' and d1['type'] == 'edge':
                if d0['op0'] == d1['op0'] and d0['op1'] == d1['op1']:
                    if not (op_types[d0['op1']] in comm_ops and op_types[d1['op1']] in comm_ops):
                        # print("non-commutative:",op_types[d0['op1']], op_types[d1['op1']])
                        if d0['port'] == d1['port']: 
                            gb.add_edge(n0, n1)
                    else:
                        # print("commutative:",op_types[d0['op1']], op_types[d1['op1']])
                        gb.add_edge(n0, n1)


    if DEBUG:
        Y = [n for n, d in gb.nodes(data=True) if d['bipartite']==0]
        nx.draw_networkx(gb, pos = nx.drawing.layout.bipartite_layout(gb, Y))
        plt.show()  

    weights = {}

    for k,v in op_types_flipped.items():
        if k == "mul":
            weights[str(v)] = 2
        else:
            weights[str(v)] = 1
        
    # print(weights)

    gc = nx.Graph()

    for i, (u, v) in enumerate(gb.edges):
        if gb.nodes.data(True)[u]['type'] == 'node':
            if gb.nodes.data(True)[u]['op'] in weights:
                weight = weights[gb.nodes.data(True)[u]['op']]
            else:
                weight = 1
        else:
            weight = 0.2
        gc.add_node(i, start=u, end=v, weight=weight)

    for pair in combinations(gc.nodes.data(True), 2):
        # print(pair)
        if ", " in pair[0][1]['start'] and ", " in pair[1][1]['start']:
            start_0_0 = pair[0][1]['start'].split(", ")[0]
            start_0_1 = pair[0][1]['start'].split(", ")[1]
            start_1_0 = pair[1][1]['start'].split(", ")[0]
            start_1_1 = pair[1][1]['start'].split(", ")[1]

            end_0_0 = pair[0][1]['end'].split(", ")[0]
            end_0_1 = pair[0][1]['end'].split(", ")[1]
            end_1_0 = pair[1][1]['end'].split(", ")[0]
            end_1_1 = pair[1][1]['end'].split(", ")[1]

            if start_0_0 == start_1_0:
                if end_0_0 == end_1_0:
                    gc.add_edge(pair[0][0], pair[1][0])
            elif start_0_0 == start_1_1:
                if end_0_0 == end_1_1:
                    gc.add_edge(pair[0][0], pair[1][0])
            elif start_0_1 == start_1_0:
                if end_0_1 == end_1_0:
                    gc.add_edge(pair[0][0], pair[1][0])
            elif start_0_1 == start_1_1:
                if end_0_1 == end_1_1:
                    gc.add_edge(pair[0][0], pair[1][0])
            elif not (end_0_0 == end_1_0 or end_0_0 == end_1_1 or end_0_1 == end_1_0 or end_0_1 == end_1_1):
                gc.add_edge(pair[0][0], pair[1][0])

        elif ", " in pair[0][1]['start']:
            start_0_0 = pair[0][1]['start'].split(", ")[0]
            start_0_1 = pair[0][1]['start'].split(", ")[1]
            start_1 = pair[1][1]['start']

            end_0_0 = pair[0][1]['end'].split(", ")[0]
            end_0_1 = pair[0][1]['end'].split(", ")[1]
            end_1 = pair[1][1]['end']

            if start_0_0 == start_1:
                if end_0_0 == end_1:
                    gc.add_edge(pair[0][0], pair[1][0])
            elif start_0_1 == start_1:
                if end_0_1 == end_1:
                    gc.add_edge(pair[0][0], pair[1][0])
            elif not (end_0_0 == end_1 or end_0_1 == end_1):
                gc.add_edge(pair[0][0], pair[1][0])

        elif ", " in pair[1][1]['start']:
            start_0 = pair[0][1]['start']
            start_1_0 = pair[1][1]['start'].split(", ")[0]
            start_1_1 = pair[1][1]['start'].split(", ")[1]

            end_0 = pair[0][1]['end']
            end_1_0 = pair[1][1]['end'].split(", ")[0]
            end_1_1 = pair[1][1]['end'].split(", ")[1]

            if start_0 == start_1_0:
                if end_0 == end_1_0:
                    gc.add_edge(pair[0][0], pair[1][0])
            elif start_0 == start_1_1:
                if end_0 == end_1_1:
                    gc.add_edge(pair[0][0], pair[1][0])
            elif not (end_0 == end_1_0 or end_0 == end_1_1):
                gc.add_edge(pair[0][0], pair[1][0])

        elif (pair[0][1]['start'] == pair[1][1]['start']):
            if (pair[0][1]['end'] == pair[1][1]['end']):
                gc.add_edge(pair[0][0], pair[1][0])
        elif not (pair[0][1]['end'] == pair[1][1]['end']):
            gc.add_edge(pair[0][0], pair[1][0])

    if DEBUG:
        starts = nx.get_node_attributes(gc, 'start') 
        ends = nx.get_node_attributes(gc, 'end') 
        labels = {n:d + "/" + ends[n] for n, d in starts.items()}
        nx.draw_networkx(gc, labels = labels)
        plt.show()  

    return gc


def FindMaximumWeightClique(gc):


    V = gc.nodes
    V = [str(v) for v in V]

    w = nx.get_node_attributes(gc, 'weight') 
    w = {str(k): v for k, v in w.items()}

    E = [e for e in gc.edges]
    E = [(str(e1), str(e2)) for (e1, e2) in E]


    def notE(V, E):
        nots = []
        for u in V:
            for v in V:
                if u ==v: continue
                if not ((u,v) in E or (v,u) in E) and not((u,v) in E or (v,u) in nots):
                    nots.append((u,v))
        return nots

    # print("Vertices und weights:", w)
    # print("Edges:               ", E)
    # print("Missing edges:       ", notE(V, E))

    model = pulp.LpProblem("max weighted clique", pulp.LpMaximize)

    xv = {}

    for v in V:
        xv[v] = pulp.LpVariable(v, lowBound=0, cat='Binary')

    model += pulp.lpSum([w[v] * xv[v] for v in V]), "max me"

    nonEdges = notE(V,E)
    for noe in nonEdges:
        model += xv[noe[0]] + xv[noe[1]] <= 1

    model.solve()
    pulp.LpStatus[model.status]

    C = []

    for v in V:
        if xv[v].varValue > 0:
            C.append((gc.nodes.data(True)[int(v)]['start'], gc.nodes.data(True)[int(v)]['end']))


    return C



def ReconstructResultingGraph(c, g1, g2):

    b = {j.replace('b',''):i.replace('a','') for (i,j) in c}

    g = g1.copy()

    idx = len(g1.nodes)
    for n, d in g2.nodes.data(True):
        if not n in b:
            g.add_node(str(idx), op=d['op'])
            b[n] = str(idx)
            idx += 1
        
    for u, v, d in g2.edges.data(True):
        if not str(u) + ", " + str(v) in b:
            g.add_edge(b[u], b[v], port=d['port'])

    if DEBUG:
        graphs = [g1, g2, g]
        for i, g in enumerate(graphs):
            plt.subplot(1, 3, i + 1)
            groups = set(nx.get_node_attributes(g,'op').values())
            mapping = dict(zip(sorted(groups),count()))
            nodes = g.nodes()
            colors = [mapping[g.node[n]['op']] for n in nodes]
            labels={}
            for n in nodes:
                labels[n] = op_types[g.node[n]['op']]

            pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
            ec = nx.draw_networkx_edges(g, pos, alpha=1, width=2)
            nc = nx.draw_networkx_nodes(g, pos, node_list = nodes, node_color=colors, 
                                        with_labels=False, node_size=500, cmap=plt.cm.Pastel1, alpha = 1)
            nx.draw_networkx_labels(g, pos, labels, font_size=8)

        plt.show()
    return g


def sort_modules(modules):
    ids = []

    output_modules = []
    while len(modules) > 0:
        for module in modules:
            if module['type'] == 'const':
                ids.append(module["id"])
                output_modules.append(module)
                modules.remove(module)

            if "in0" in module and "in1" in module:
                if "in" in module["in0"] and "in" in module["in1"]:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)
                elif module["in0"] in ids and "in" in module["in1"]:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)
                elif "in" in module["in0"] and module["in1"] in ids:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)
                elif module["in0"] in ids and module["in1"] in ids:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)

    return output_modules

def merged_subgraph_to_arch(subgraph):

    # pdb.set_trace()

    with open(".temp/op_types.txt") as file:
        op_types = ast.literal_eval(file.read())

    op_types = {str(v): k for k, v in op_types.items()}

    alu_supported_ops =  ["not", "and", "or", "xor", "shl", "lshr", "ashr", "neg", "add", "sub", "sle", "sge", "ule", "uge", "eq", "mux", "slt", "sgt", "ult", "ugt"]
    
    arch = {}
    arch["input_width"] = 16
    arch["output_width"] = 16
    arch["enable_input_regs"] = False
    arch["enable_output_regs"] = False
    modules = {}
    ids = []
    connected_ids = []

    for n, d in subgraph.nodes.data(True):
        # print("n=" + n + " d=" + d)
        modules[n] = {}
        modules[n]["id"] = n

        op = op_types[d["op"]]

        if op == "mul" or op == "const":
            modules[n]["type"] = op
        else:
            if not op in alu_supported_ops:
                print("Warning: possible unsupported ALU operation found in subgraph:", op)
            modules[n]["type"] = op.replace(op,'alu')      

        ids.append(n)

    for u, v, d in subgraph.edges.data(True):
        connected_ids.append(u)

        if d["port"] == "0":
            if "in0" in modules[v]:
                modules[v]["in0"].append(u)
            else:
                modules[v]["in0"] = [u]
        else:
            if "in1" in modules[v]:
                modules[v]["in1"].append(u)
            else:
                modules[v]["in1"] = [u]

    arch["modules"] = [v for v in modules.values()]
    arch["outputs"] = [i for i in ids if i not in connected_ids]


    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    input_counter = 0
    for module in arch["modules"]:
        if not module['type'] == 'const':
            if 'in0' not in module:
                module["in0"] = "in" + str(input_counter)
                input_counter += 1
            if 'in1' not in module:
                module["in1"] = "in" + str(input_counter)
                input_counter += 1

    arch["modules"] = sort_modules(arch["modules"])

    with open("outputs/subgraph_arch_merged.json", "w") as write_file:
        write_file.write(json.dumps(arch, indent = 4, sort_keys=True))


# with open(".temp/grami_in.txt") as file:
#     lines = file.readlines()

# graph = nx.DiGraph()
# for line in lines:
#     if 'v' in line:
#         graph.add_node(line.split()[1], type=line.split()[2])
#     elif 'e' in line:
#         graph.add_edge(line.split()[1], line.split()[2], port=line.split()[3])

DEBUG = False

with open(".temp/grami_out.txt") as file:
    lines = file.readlines()

graph_ind = -1
graphs = []

for line in lines:
    if ':' in line:
        graph_ind += 1
        graphs.append(nx.DiGraph())
    elif 'v' in line:
        graphs[graph_ind].add_node(line.split()[1], op=line.split()[2])
    elif 'e' in line:
        graphs[graph_ind].add_edge(line.split()[1], line.split()[2], port=line.split()[3])


with open(".temp/op_types.txt") as file:
    op_types_flipped = ast.literal_eval(file.read())

op_types = {str(v): k for k, v in op_types_flipped.items()}

# graphs.reverse()

G = graphs[0]

for i in range(1, len(graphs)):
    gc = ConstructCompatibilityGraph(G, graphs[i])

    C = FindMaximumWeightClique(gc)

    G = ReconstructResultingGraph(C, G, graphs[i])


# graphs.append(G)

# for i, g in enumerate(graphs):
#     plt.subplot(math.ceil(len(graphs)/7), 7, i + 1)
#     groups = set(nx.get_node_attributes(g,'op').values())
#     mapping = dict(zip(sorted(groups),count()))
#     nodes = g.nodes()
#     colors = [mapping[g.node[n]['op']] for n in nodes]
#     labels={}
#     for n in nodes:
#         labels[n] = op_types[g.node[n]['op']]

#     pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
#     ec = nx.draw_networkx_edges(g, pos, alpha=1, width=2)
#     nc = nx.draw_networkx_nodes(g, pos, node_list = nodes, node_color=colors, 
#                                 with_labels=False, node_size=500, cmap=plt.cm.Pastel1, alpha = 1)
#     nx.draw_networkx_labels(g, pos, labels, font_size=8)

# plt.show()

g = G
groups = set(nx.get_node_attributes(g,'op').values())
mapping = dict(zip(sorted(groups),count()))
nodes = g.nodes()
colors = [mapping[g.node[n]['op']] for n in nodes]
labels={}
for n in nodes:
    labels[n] = op_types[g.node[n]['op']]

pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
ec = nx.draw_networkx_edges(g, pos, alpha=1, width=2)
nc = nx.draw_networkx_nodes(g, pos, node_list = nodes, node_color=colors, 
                            with_labels=False, node_size=500, cmap=plt.cm.Pastel1, alpha = 1)
nx.draw_networkx_labels(g, pos, labels, font_size=8)

plt.show()

merged_subgraph_to_arch(g)