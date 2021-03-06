from itertools import count, combinations
from networkx.drawing import nx_agraph
from networkx.algorithms import bipartite

import networkx as nx
import pulp
import ast
import os
import copy
import subgraph_merging.config as config

from .utils import *
from .rewrite_rule_utils import *
from .plot_utils import *

def add_input_and_output_nodes(g):

    input_idx = 0
    bit_input_idx = 0
    const_idx = 0
    bit_const_idx = 0

    for n, d in g.copy().nodes.data(True):
        if not config.op_types[d["op"]] == "const" and not config.op_types[d["op"]] == "bitconst":
            pred = {}
            for s in g.pred[n]:
                pred[g.edges[(s, n, 0)]['port']] = s
            if '0' not in pred:
                if config.op_types[d["alu_ops"][0]] in config.bit_input_ops:
                    g.add_node("bit_in" + str(bit_input_idx), op="bit_input", alu_ops=["bit_input"])
                    g.add_edge("bit_in" + str(bit_input_idx), n, port='0')
                    bit_input_idx += 1                    
                else:
                    g.add_node("in" + str(input_idx), op="input", alu_ops=["input"])
                    g.add_edge("in" + str(input_idx), n, port='0')
                    input_idx += 1
            if '1' not in pred:
                if config.op_types[d["alu_ops"][0]] in config.bit_input_ops:
                    g.add_node("bit_in" + str(bit_input_idx), op="bit_input", alu_ops=["bit_input"])
                    g.add_edge("bit_in" + str(bit_input_idx), n, port='1')
                    bit_input_idx += 1                    
                else:
                    g.add_node("in" + str(input_idx), op="input", alu_ops=["input"])
                    g.add_edge("in" + str(input_idx), n, port='1')
                    input_idx += 1
            if config.op_types[d["alu_ops"][0]] in config.bit_input_ops or config.op_types[d["alu_ops"][0]] == "mux":
                if '2' not in pred:
                    g.add_node("bit_in" + str(bit_input_idx), op="bit_input", alu_ops=["bit_input"])
                    g.add_edge("bit_in" + str(bit_input_idx), n, port='2')
                    bit_input_idx += 1
        else:
            if config.op_types[d["op"]] == "const":
                g.add_node(
                    "const" + str(const_idx),
                    op="const_input",
                    alu_ops=["const_input"])
                g.add_edge("const" + str(const_idx), n, port='0')
                const_idx += 1
            else:
                g.add_node(
                    "bit_const" + str(bit_const_idx),
                    op="bit_const_input",
                    alu_ops=["bit_const_input"])
                g.add_edge("bit_const" + str(bit_const_idx), n, port='0')
                bit_const_idx += 1
    for n, d in g.copy().nodes.data(True):
        if len(list(g.successors(n))) == 0:
            if config.op_types[d["alu_ops"][0]] in config.bit_output_ops:
                g.add_node("bit_out0", op="bit_output", alu_ops=["bit_output"])
                g.add_edge(n, "bit_out0", port='0')
            else:
                g.add_node("out0", op="output", alu_ops=["output"])
                g.add_edge(n, "out0", port='0')


# def RemoveInputAndOutputNodes(g):
#     ret_g = g.copy()
#     for n, d in ret_g.copy().nodes.data(True):
#         if d["op"] == "input" or d["op"] == "input" or d["op"] == "const_input":
#             ret_g.remove_node(n)

#     ret_g.remove_node("out0")
#     return ret_g


# config.comm_ops =  []
def construct_compatibility_graph(g1, g2):

    gb = nx.Graph()
    g1_map = {}
    g1_map_r = {}
    idx = 0
    for n1 in g1.nodes():
        gb.add_node(
            'a' + str(idx),
            type='node',
            op=g1.nodes[n1]['op'],
            alu_ops=g1.nodes[n1]['alu_ops'],
            bipartite=0)
        g1_map[n1] = 'a' + str(idx)
        g1_map_r['a' + str(idx)] = n1
        idx += 1

    for (u, v, p) in g1.edges.data('port'):
        gb.add_node(
            g1_map[u] + ', ' + g1_map[v],
            type='edge',
            node0=g1_map[u],
            node1=g1_map[v],
            op0=g1.nodes[u]['op'],
            op1=g1.nodes[v]['op'],
            alu_ops1=g1.nodes[v]['alu_ops'],
            port=p,
            bipartite=0)

    g2_map = {}
    g2_map_r = {}
    idx = 0
    for n1 in g2.nodes():
        gb.add_node(
            'b' + str(idx),
            type='node',
            op=g2.nodes[n1]['op'],
            alu_ops=g2.nodes[n1]['alu_ops'],
            bipartite=1)
        g2_map[n1] = 'b' + str(idx)
        g2_map_r['b' + str(idx)] = n1
        idx += 1

    for (u, v, p) in g2.edges.data('port'):
        gb.add_node(
            g2_map[u] + ', ' + g2_map[v],
            type='edge',
            node0=g2_map[u],
            node1=g2_map[v],
            op0=g2.nodes[u]['op'],
            op1=g2.nodes[v]['op'],
            alu_ops1=g2.nodes[v]['alu_ops'],
            port=p,
            bipartite=1)


    left_nodes = [(n, d) for n, d in gb.nodes(data=True)
                  if d['bipartite'] == 0]
    right_nodes = [(n, d) for n, d in gb.nodes(data=True)
                   if d['bipartite'] == 1]

    for n0, d0 in left_nodes:
        for n1, d1 in right_nodes:
            if d0['type'] == 'node' and d1['type'] == 'node':
                # print(d0['op'], " -> ", d1['op'])
                if d0['op'] == d1['op']:
                    gb.add_edge(n0, n1)
            elif d0['type'] == 'edge' and d1['type'] == 'edge':
                if d0['op0'] == d1['op0'] and d0['op1'] == d1['op1']:
                    if d0['port'] == d1['port']:
                        gb.add_edge(n0, n1)
                    else:
                        if config.op_types[d1['alu_ops1'][0]] in config.comm_ops:
                            gb.add_edge(n0, n1)

    weights = {}

    for k, v in config.op_types_flipped.items():
        if k == "mul":
            weights[str(v)] = 2030
        elif k == "const":
            weights[str(v)] = 12
        elif k == "mux":
            weights[str(v)] = 30
        elif k == "lut":
            weights[str(v)] = 13
        else:
            weights[str(v)] = 1317

    gc = nx.Graph()

    for i, (u, v) in enumerate(gb.edges):
        in_or_out = False
        if gb.nodes.data(True)[u]['type'] == 'node':
            if gb.nodes.data(True)[u]['op'] in weights:
                weight = weights[gb.nodes.data(True)[u]['op']]
            else:
                weight = 1317
            if gb.nodes.data(True)[u]['op'] == "input" or gb.nodes.data(True)[u]['op'] == "bit_input" \
                or gb.nodes.data(True)[u]['op'] == "output" or gb.nodes.data(True)[u]['op'] == "bit_output" or gb.nodes.data(True)[v]['op'] == "bit_output" or gb.nodes.data(True)[u]['op'] == "const_input" or gb.nodes.data(True)[u]['op'] == "bit_const_input":
                in_or_out = True
            gc.add_node(i, start=u, end=v, weight=weight, in_or_out=in_or_out)
        else:
            weight = 30
            d = gb.nodes.data(True)[u]
            if d["op0"] == "input" or d["op1"] == "input" or d["op0"] == "bit_input" or d["op1"] == "bit_input" or d["op0"] == "output" \
                or d["op1"] == "output" or d["op0"] == "bit_output" or d["op1"] == "bit_output"or d["op0"] == "const_input" or d["op1"] == "const_input" or d["op0"] == "bit_const_input" or d["op1"] == "bit_const_input":
                in_or_out = True
            gc.add_node(i, start=u, end=v, weight=weight, in_or_out=in_or_out, start_port=gb.nodes.data(True)[u]["port"], end_port=gb.nodes.data(True)[v]["port"])

    for pair in combinations(gc.nodes.data(True), 2):
        
        if ", " in pair[0][1]['start'] and ", " in pair[1][1]['start']:
            start_0_0 = pair[0][1]['start'].split(", ")[0]
            start_0_1 = pair[0][1]['start'].split(", ")[1]
            start_1_0 = pair[1][1]['start'].split(", ")[0]
            start_1_1 = pair[1][1]['start'].split(", ")[1]

            end_0_0 = pair[0][1]['end'].split(", ")[0]
            end_0_1 = pair[0][1]['end'].split(", ")[1]
            end_1_0 = pair[1][1]['end'].split(", ")[0]
            end_1_1 = pair[1][1]['end'].split(", ")[1]

            start_0_1_port = pair[0][1]['start_port']
            start_1_1_port = pair[1][1]['start_port']
            end_0_1_port = pair[0][1]['end_port']
            end_1_1_port = pair[1][1]['end_port']

            # (a0, a1) -> (b0, b1)
            # (a2, a3) -> (b2, b3)

            # (start_0_0, start_0_1) -> (end_0_0, end_0_1)
            # (start_1_0, start_1_1) -> (end_1_0, end_1_1)

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
                    if start_0_1_port != start_1_1_port and end_0_1_port != end_1_1_port:
                        gc.add_edge(pair[0][0], pair[1][0])
            elif not (end_0_0 == end_1_0 or end_0_0 == end_1_1
                      or end_0_1 == end_1_0 or end_0_1 == end_1_1):
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
                if check_no_cycles(pair, g1, g1_map_r, g2, g2_map_r):
                    gc.add_edge(pair[0][0], pair[1][0])
        elif not (pair[0][1]['end'] == pair[1][1]['end']):
            if check_no_cycles(pair, g1, g1_map_r, g2, g2_map_r):
                gc.add_edge(pair[0][0], pair[1][0])

    if DEBUG:
        plot_compatibility_graph(g1, g1_map, g2, g2_map, gb, gc, config.op_types)

    return gc, g1_map_r, g2_map_r


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
                if u == v:
                    continue
                if not ((u, v) in E or (v, u) in E) and not ((u, v) in E or
                                                             (v, u) in nots):
                    nots.append((u, v))
        return nots

    model = pulp.LpProblem("max weighted clique", pulp.LpMaximize)

    xv = {}

    for v in V:
        xv[v] = pulp.LpVariable(v, lowBound=0, cat='Binary')

    model += pulp.lpSum([w[v] * xv[v] for v in V]), "max me"

    nonEdges = notE(V, E)
    for noe in nonEdges:
        model += xv[noe[0]] + xv[noe[1]] <= 1

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    pulp.LpStatus[model.status]

    C = []
    widths = []
    for v in V:
        if xv[v].varValue > 0:
            C.append((gc.nodes.data(True)[int(v)]['start'],
                      gc.nodes.data(True)[int(v)]['end']))
            widths.append(0)
        else:
            widths.append(2)

    if DEBUG:
        plot_max_weight_clique(gc, widths)


    return C

def swap_ports(subgraph, dest_node):
    for s in subgraph.pred[dest_node]:
        if subgraph.edges[(s, dest_node, 0)]['port'] == "0":
            subgraph.edges[(s, dest_node, 0)]['port'] = "1"
        else:
            subgraph.edges[(s, dest_node, 0)]['port'] = "0"

        print(s, subgraph.nodes.data(True)[s], dest_node, subgraph.nodes.data(True)[dest_node], subgraph.edges[(s, dest_node, 0)])

def reconsruct_resulting_graph(c, g1, g2, g1_map, g2_map):
    b = {}

    for (i, j) in c:
        if len(i.split(", ")) > 1:
            b[g2_map[j.split(", ")[0]] + ", " +
              g2_map[j.split(", ")[1]]] = g1_map[i.split(
                  ", ")[0]] + ", " + g1_map[i.split(", ")[1]]
        else:
            b[g2_map[j]] = g1_map[i]


    # breakpoint()
    for k,v in b.items():
        if "," in k:
            g2u = k.split(", ")[0]
            g2v = k.split(", ")[1]
            g1u = v.split(", ")[0]
            g1v = v.split(", ")[1]
            if g1.edges[(g1u, g1v, 0)]['port'] != g2.edges[(g2u, g2v, 0)]['port'] and (g1u, g1v, 1) not in g1.edges:
                if config.op_types[g2.nodes.data(True)[g2v]['alu_ops'][0]] in config.comm_ops:
                    swap_ports(g2, g2v)
                else:
                    print("Oops, something went wrong")
                    exit()

    g = g1.copy()

    in_idx = 0
    bit_in_idx = 0
    const_idx = 0
    bit_const_idx = 0
    output_idx = 0
    bit_output_idx = 0

    for n in g1.nodes:
        if "in" in n and "bit" not in n:
            in_idx += 1
        if "in" in n and "bit" in n:
            bit_in_idx += 1
        if "const" in n:
            const_idx += 1
        if "bit_const" in n:
            bit_const_idx += 1
        if "output_idx" in n:
            output_idx += 1
        if "bit_output_idx" in n:
            bit_output_idx += 1


    idx = len(g1.nodes)
    for n, d in g2.nodes.data(True):
        if n not in b:
            if d["op"] == "input":
                g.add_node("in" + str(in_idx), op=d['op'], alu_ops=d['alu_ops'])
                b[n] = "in" + str(in_idx)
                in_idx += 1
            elif d["op"] == "bit_input":
                g.add_node("bit_in" + str(bit_in_idx), op=d['op'], alu_ops=d['alu_ops'])
                b[n] = "bit_in" + str(bit_in_idx)
                bit_in_idx += 1
            elif d["op"] == "const_input":
                g.add_node(
                    "const" + str(const_idx), op=d['op'], alu_ops=d['alu_ops'])
                b[n] = "const" + str(const_idx)
                const_idx += 1
            elif d["op"] == "bit_const_input":
                g.add_node(
                    "bit_const" + str(bit_const_idx), op=d['op'], alu_ops=d['alu_ops'])
                b[n] = "bit_const" + str(bit_const_idx)
                bit_const_idx += 1
            elif d["op"] == "output":
                g.add_node(
                    "out" + str(output_idx), op=d['op'], alu_ops=d['alu_ops'])
                b[n] = "out" + str(output_idx)
                output_idx += 1
            elif d["op"] == "bit_output":
                g.add_node(
                    "bit_out" + str(bit_output_idx), op=d['op'], alu_ops=d['alu_ops'])
                b[n] = "bit_out" + str(bit_output_idx)
                bit_output_idx += 1
            else:
                g.add_node(str(idx), op=d['op'], alu_ops=d['alu_ops'])
                b[n] = str(idx)
                idx += 1
        else:
            if d["op"] == "alu":
                g1.nodes.data(True)[b[n]]['alu_ops'] += d['alu_ops']

    # breakpoint()
    for u, v, d in g2.edges.data(True):
        if not str(u) + ", " + str(v) in b and not g.has_edge(b[u],b[v]):
            g.add_edge(b[u], b[v], port=d['port'])


    if DEBUG:
        plot_reconstructed_graph(g1, g2, g, config.op_types)

    return g, b

def merge_subgraphs(file_ind_pairs):
    clean_output_dirs()

    with open(".temp/op_types.txt", "rb") as file:
        op_types_from_file = pickle.load(file)

    curr_ops = [*op_types_from_file]

    for op in config.primitive_ops:
        if op not in curr_ops:
            curr_ops.append(op)

    config.op_types = {str(k): v for k, v in enumerate(curr_ops)}

    for op in config.non_coreir_ops:
        if op not in config.op_types:
            config.op_types[op] = op

    config.op_types_flipped = {v: k for k, v in config.op_types.items()}

    graphs = read_subgraphs(file_ind_pairs)

    add_primitive_ops(graphs)

    for graph in graphs:
        add_input_and_output_nodes(graph)

    rrules = []

    G = graphs[0]
    mappings = []

    for i in range(1, len(graphs)):
        gc, g1_map, g2_map = construct_compatibility_graph(G, graphs[i])

        C = FindMaximumWeightClique(gc)

        G, new_mapping = reconsruct_resulting_graph(C, G, graphs[i], g1_map, g2_map)

        if i == 1:
            print("subgraph ", 0)
            new_mapping_0 = {k:k for k in graphs[0].nodes}
            new_mapping_0.update({u + ", " + v:u + ", " + v for (u,v,z) in graphs[0].edges})
            node_dict = subgraph_to_peak(graphs[0], 0, new_mapping_0)
            rrules.append(
                gen_rewrite_rule(node_dict))
            mappings.append(new_mapping_0)
            print()

        print("subgraph ", i)
        mappings.append(new_mapping)
        node_dict = subgraph_to_peak(graphs[i], i, new_mapping)
        rrules.append(gen_rewrite_rule(node_dict))
        print()


    plot_graph(G)
    merged_arch = merged_subgraph_to_arch(G)
    constraints = formulate_rewrite_rules(rrules, merged_arch)

    rrules = test_rewrite_rules(constraints)

    write_rewrite_rules(rrules)

    gen_verilog()
    
DEBUG = False