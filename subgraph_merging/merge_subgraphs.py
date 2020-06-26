from itertools import count, combinations
from networkx.drawing import nx_agraph
from networkx.algorithms import bipartite

import networkx as nx
import matplotlib.pyplot as plt
import pulp
import sys
import ast
import math
import os
import json
import copy
import importlib
import time

from hwtypes import BitVector, Tuple, Bit
from peak import family
from peak import Peak, family_closure
from peak.mapper import ArchMapper
from peak.mapper.utils import pretty_print_binding
from peak.family import AbstractFamily
from peak.mapper import RewriteRule
from peak.assembler.assembled_adt import  AssembledADT
from peak.assembler.assembler import Assembler
from peak_gen.sim import wrapped_pe_arch_closure
from peak_gen.arch import read_arch, graph_arch
from peak_gen.isa import inst_arch_closure
from peak_gen.asm import asm_arch_closure
from peak_gen.alu import ALU_t, Signed_t
from peak_gen.mul import MUL_t
import peak
import magma as m
import shutil 


def add_input_and_output_nodes(g, op_types):
    node_dict = {}
    input_idx = 0
    const_idx = 0

    for n, d in g.copy().nodes.data(True):
        if not op_types[d["op"]] == "const":
            pred = {}
            for s in g.pred[n]:
                pred[g.edges[(s, n, 0)]['port']] = s
            if '0' not in pred:
                g.add_node("in" + str(input_idx), op="input", alu_op="input")
                g.add_edge("in" + str(input_idx), n, port='0')
                input_idx += 1
            if '1' not in pred:
                g.add_node("in" + str(input_idx), op="input", alu_op="input")
                g.add_edge("in" + str(input_idx), n, port='1')
                input_idx += 1
        else:
            g.add_node(
                "const" + str(const_idx),
                op="const_input",
                alu_op="const_input")
            g.add_edge("const" + str(const_idx), n, port='0')
            const_idx += 1
    for n in g.copy().nodes:
        if len(list(g.successors(n))) == 0:
            g.add_node("out0", op="output", alu_op="output")
            g.add_edge(n, "out0", port='0')


def RemoveInputAndOutputNodes(g):
    ret_g = g.copy()
    for n, d in ret_g.copy().nodes.data(True):
        if d["op"] == "input" or d["op"] == "const_input":
            ret_g.remove_node(n)
        elif d["op"] == "const_input":
            d.pop('0', None)
    ret_g.remove_node("out0")
    return ret_g



def construct_compatibility_graph(g1, g2, op_types, op_types_flipped):

    gb = nx.Graph()
    g1_map = {}
    g1_map_r = {}
    idx = 0
    for n1 in g1.nodes():
        gb.add_node(
            'a' + str(idx),
            type='node',
            op=g1.node[n1]['op'],
            alu_op=g1.node[n1]['alu_op'],
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
            op0=g1.node[u]['op'],
            op1=g1.node[v]['op'],
            alu_op1=g1.node[v]['alu_op'],
            port=p,
            bipartite=0)

    g2_map = {}
    g2_map_r = {}
    idx = 0
    for n1 in g2.nodes():
        gb.add_node(
            'b' + str(idx),
            type='node',
            op=g2.node[n1]['op'],
            alu_op=g2.node[n1]['alu_op'],
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
            op0=g2.node[u]['op'],
            op1=g2.node[v]['op'],
            alu_op1=g2.node[v]['alu_op'],
            port=p,
            bipartite=1)

    comm_ops =  ["and", "or", "xor", "add", "eq", "mul", "alu"]
    # comm_ops = []
    left_nodes = [(n, d) for n, d in gb.nodes(data=True)
                  if d['bipartite'] == 0]
    right_nodes = [(n, d) for n, d in gb.nodes(data=True)
                   if d['bipartite'] == 1]

    for n0, d0 in left_nodes:
        for n1, d1 in right_nodes:
            if d0['type'] == 'node' and d1['type'] == 'node':
                if d0['op'] == d1['op']:
                    gb.add_edge(n0, n1)
            elif d0['type'] == 'edge' and d1['type'] == 'edge':
                if d0['op0'] == d1['op0'] and d0['op1'] == d1['op1']:
                    if d0['port'] == d1['port']:
                        gb.add_edge(n0, n1)
                    else:
                        if op_types[d1['alu_op1']] in comm_ops or op_types[d0['alu_op1']] in comm_ops:
                            gb.add_edge(n0, n1)

            
    if DEBUG:

        plt.subplot(1, 4, 1)
        plt.margins(0.2)
        g = g1
        groups1 = set(nx.get_node_attributes(g1, 'op').values())
        groups2 = set(nx.get_node_attributes(g2, 'op').values())
        groups = groups1.union(groups2)
        mapping = dict(zip(sorted(groups), count()))
        nodes = g.nodes()
        colors = [plt.cm.Pastel1(mapping[g.node[n]['op']]) for n in nodes]

        labels = {}
        for n in nodes:
            labels[n] = g1_map[n] + "\n" + op_types[g.node[n]['op']]
        pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
        ec = nx.draw_networkx_edges(
            g,
            pos,
            alpha=1,
            width=3,
            node_size=1500,
            arrows=True,
            arrowsize=15)
        nc = nx.draw_networkx_nodes(
            g,
            pos,
            node_list=nodes, 
            node_color=colors,
            with_labels=False,
            node_size=1500,
            alpha=1)
        nx.draw_networkx_labels(g, pos, labels)

        plt.subplot(1, 4, 2)
        plt.margins(0.2)
        g = g2
        nodes = g.nodes()
        colors = [plt.cm.Pastel1(mapping[g.node[n]['op']]) for n in nodes]

        labels = {}
        for n in nodes:
            labels[n] = g2_map[n] + "\n" + op_types[g.node[n]['op']]
        pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
        ec = nx.draw_networkx_edges(
            g,
            pos,
            alpha=1,
            width=3,
            node_size=1500,
            arrows=True,
            arrowsize=15)
        nc = nx.draw_networkx_nodes(
            g,
            pos,
            node_list=nodes,
            node_color=colors,
            with_labels=False,
            node_size=1500,
            alpha=1)
        nx.draw_networkx_labels(g, pos, labels)

        plot_gb = gb.copy()

        # for n, d in plot_gb.copy().nodes.data(True):
        #     if "op" in d:
        #         if d["op"] == "input" or d["op"] == "output" or d["op"] == "const_input":
        #             plot_gb.remove_node(n)
        #     else:
        #         if d["op0"] == "input" or d["op1"] == "input" or d["op0"] == "output" or d["op1"] == "output" or d["op0"] == "const_input" or d["op1"] == "const_input":
        #             plot_gb.remove_node(n)

        plt.subplot(1, 4, 3)

        left = [n for n, d in plot_gb.nodes(data=True) if d['bipartite'] == 0]
        right = [n for n, d in plot_gb.nodes(data=True) if d['bipartite'] == 1]
        pos = {}

        # Update position for node from each group
        pos.update((node, (1, index)) for index, node in enumerate(left))
        pos.update((node, (2, index)) for index, node in enumerate(right))

        nx.draw_networkx(
            plot_gb,
            node_color=plt.cm.Pastel1(2),
            node_size=1500,
            pos=pos,
            width=3)
        plt.margins(0.2)

    weights = {}

    for k, v in op_types_flipped.items():
        if k == "mul":
            weights[str(v)] = 2030
        elif k == "const":
            weights[str(v)] = 12
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
            if gb.nodes.data(True)[u]['op'] == "input" or gb.nodes.data(
                    True)[u]['op'] == "output" or gb.nodes.data(
                        True)[u]['op'] == "const_input":
                in_or_out = True
            gc.add_node(i, start=u, end=v, weight=weight, in_or_out=in_or_out)
        else:
            weight = 30
            d = gb.nodes.data(True)[u]
            if d["op0"] == "input" or d["op1"] == "input" or d["op0"] == "output" or d["op1"] == "output" or d["op0"] == "const_input" or d["op1"] == "const_input":
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
                    if start_0_1_port == end_0_1_port and end_1_1_port == start_1_1_port:
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
                gc.add_edge(pair[0][0], pair[1][0])
        elif not (pair[0][1]['end'] == pair[1][1]['end']):
            gc.add_edge(pair[0][0], pair[1][0])

    if DEBUG:

        plot_gc = gc.copy()

        # for n, d in plot_gc.copy().nodes.data(True):
        #     if d["in_or_out"]:
        #         plot_gc.remove_node(n)

        plt.subplot(1, 4, 4)
        starts = nx.get_node_attributes(plot_gc, 'start')
        ends = nx.get_node_attributes(plot_gc, 'end')
        weights = nx.get_node_attributes(plot_gc, 'weight')
        labels = {
            n: d + "/" + ends[n] + "\n" + str(weights[n])
            for n, d in starts.items()
        }
        nx.draw_networkx(
            plot_gc,
            node_color=plt.cm.Pastel1(2),
            node_size=1500,
            width=3,
            labels=labels)
        plt.margins(0.2)
        plt.show()

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

    model.solve()
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

        plot_gc = gc.copy()

        for n, d in plot_gc.copy().nodes.data(True):
            if d["in_or_out"]:
                plot_gc.remove_node(n)

        plt.subplot(1, 2, 1)
        starts = nx.get_node_attributes(plot_gc, 'start')
        ends = nx.get_node_attributes(plot_gc, 'end')
        weights = nx.get_node_attributes(plot_gc, 'weight')
        labels = {
            n: d + "/" + ends[n] + "\n" + str(weights[n])
            for n, d in starts.items()
        }
        pos = nx.drawing.layout.spring_layout(plot_gc)
        nx.draw_networkx(
            plot_gc,
            pos,
            node_color=plt.cm.Pastel1(2),
            node_size=1500,
            width=3,
            labels=labels)
        plt.margins(0.2)

        plt.subplot(1, 2, 2)
        starts = nx.get_node_attributes(plot_gc, 'start')
        ends = nx.get_node_attributes(plot_gc, 'end')
        weights = nx.get_node_attributes(plot_gc, 'weight')
        labels = {
            n: d + "/" + ends[n] + "\n" + str(weights[n])
            for n, d in starts.items()
        }

        colors = [plt.cm.Pastel1(widths[n]) for n in plot_gc.nodes()]

        nx.draw_networkx(
            plot_gc,
            pos,
            node_color=colors,
            node_size=1500,
            width=3,
            labels=labels)
        plt.margins(0.2)
        plt.show()

    return C

def swap_ports(subgraph, dest_node):
    # Find all predecessors of dest_node in subgraph
    for s in subgraph.pred[dest_node]:
        if subgraph.edges[(s, dest_node, 0)]['port'] == "0":
            subgraph.edges[(s, dest_node, 0)]['port'] = "1"
        else:
            subgraph.edges[(s, dest_node, 0)]['port'] = "0"

        # print(s, subgraph.nodes.data(True)[s], dest_node, subgraph.nodes.data(True)[dest_node], subgraph.edges[(s, dest_node, 0)])

def reconsruct_resulting_graph(c, g1, g2, g1_map, g2_map, op_types):
    b = {}


    for (i, j) in c:
        if len(i.split(", ")) > 1:
            b[g2_map[j.split(", ")[0]] + ", " +
              g2_map[j.split(", ")[1]]] = g1_map[i.split(
                  ", ")[0]] + ", " + g1_map[i.split(", ")[1]]
        else:
            b[g2_map[j]] = g1_map[i]

    comm_ops =  ["and", "or", "xor", "add", "eq", "mul", "alu"]

    for k,v in b.items():
        if "," in k:
            g2u = k.split(", ")[0]
            g2v = k.split(", ")[1]
            g1u = v.split(", ")[0]
            g1v = v.split(", ")[1]
            if g1.edges[(g1u, g1v, 0)]['port'] != g2.edges[(g2u, g2v, 0)]['port'] and (g1u, g1v, 1) not in g1.edges:
                if op_types[g2.nodes.data(True)[g2v]['alu_op']] in comm_ops:
                    swap_ports(g2, g2v)
                elif op_types[g1.nodes.data(True)[g1v]['alu_op']] in comm_ops:
                    swap_ports(g1, g1v)
                    breakpoint()
                
                else:
                    print("Oops, something went wrong")
                    exit()

    g = g1.copy()

    in_idx = 0
    for n in g1.nodes:
        if "in" in n:
            in_idx += 1

    const_idx = 0
    for n in g1.nodes:
        if "const" in n:
            const_idx += 1

    idx = len(g1.nodes)
    for n, d in g2.nodes.data(True):
        if n not in b:
            if d["op"] == "input":
                g.add_node("in" + str(in_idx), op=d['op'], alu_op=d['alu_op'])
                b[n] = "in" + str(in_idx)
                in_idx += 1
            elif d["op"] == "const_input":
                g.add_node(
                    "const" + str(const_idx), op=d['op'], alu_op=d['alu_op'])
                b[n] = "const" + str(const_idx)
                const_idx += 1
            else:
                g.add_node(str(idx), op=d['op'], alu_op=d['alu_op'])
                b[n] = str(idx)
                idx += 1


    for u, v, d in g2.edges.data(True):
        if not str(u) + ", " + str(v) in b:
            g.add_edge(b[u], b[v], port=d['port'])


    if DEBUG:
        graphs = [g1, g2, g]
        for i, g in enumerate(graphs):

            ret_g = g.copy()



            plt.subplot(1, 3, i + 1)
            groups = set(nx.get_node_attributes(ret_g, 'op').values())
            mapping = dict(zip(sorted(groups), count()))
            nodes = ret_g.nodes()
            colors = [mapping[ret_g.node[n]['op']] for n in nodes]
            labels = {}
            for n in nodes:
                labels[n] = op_types[ret_g.node[n]['op']] + "\n" + n

            pos = nx.nx_agraph.graphviz_layout(ret_g, prog='dot')
            ec = nx.draw_networkx_edges(
                ret_g,
                pos,
                alpha=1,
                width=3,
                node_size=1500,
                arrows=True,
                arrowsize=15)
            nc = nx.draw_networkx_nodes(
                ret_g,
                pos,
                node_list=nodes,
                node_color=colors,
                with_labels=False,
                node_size=1500,
                cmap=plt.cm.Pastel1,
                alpha=1)
            nx.draw_networkx_labels(ret_g, pos, labels)

        plt.show()

    return g, b


def sort_modules(modules):
    ids = []

    output_modules = []
    while len(modules) > 0:
        for module in modules.copy():
            if module['type'] == 'const':
                ids.append(module["id"])
                output_modules.append(module)
                modules.remove(module)

            if "in0" in module and "in1" in module:
                inorder = True

                for in0_item in module["in0"]:
                    inorder = inorder and ("in" in in0_item or in0_item in ids)

                for in1_item in module["in1"]:
                    inorder = inorder and ("in" in in1_item or in1_item in ids)

                if inorder:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)

    return output_modules


def merged_subgraph_to_arch(subgraph, op_types):

    alu_supported_ops = [
        "alu", "not", "and", "or", "xor", "shl", "lshr", "ashr", "neg", "add",
        "sub", "sle", "sge", "ule", "uge", "eq", "mux", "slt", "sgt", "ult",
        "ugt"
    ]

    bit_output_ops = [
        "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt"
    ]

    arch = {}
    arch["input_width"] = 16
    arch["output_width"] = 16
    arch["enable_input_regs"] = False
    arch["enable_output_regs"] = False
    modules = {}
    ids = []
    connected_ids = []
    bit_outputs = []

    for n, d in subgraph.nodes.data(True):

        # Don't want to put input and output nodes in actual arch
        if d["op"] != "output" and d["op"] != "const_input":

            # Only want id of input nodes
            if d["op"] != "input":
                modules[n] = {}
                modules[n]["id"] = n

                op = op_types[d["op"]]

                if op == "mul" or op == "const":
                    modules[n]["type"] = op
                else:
                    if op not in alu_supported_ops:
                        print(
                            "Warning: possible unsupported ALU operation found in subgraph:",
                            op)
                    modules[n]["type"] = op.replace(op, 'alu')

                if op in bit_output_ops:
                    bit_outputs.append(n)

            ids.append(n)

    outputs = set()

    for u, v, d in subgraph.edges.data(True):
        if v in modules:
            connected_ids.append(u)
            if modules[v]["type"] != "const":
                if d["port"] == "0":
                    if "in0" in modules[v]:
                        modules[v]["in0"].append(u)
                    else:
                        modules[v]["in0"] = [u]
                elif d["port"] == "1":
                    if "in1" in modules[v]:
                        modules[v]["in1"].append(u)
                    else:
                        modules[v]["in1"] = [u]
        
        # Add to output mux
        if subgraph.nodes.data(True)[v]["op"] == "output":
            outputs.add(u)


    arch["modules"] = [v for v in modules.values()]
    arch["outputs"] = [list(outputs)]
    arch["bit_outputs"] = [i for i in bit_outputs if i not in connected_ids]

    if not os.path.exists('outputs/subgraph_archs'):
        os.makedirs('outputs/subgraph_archs')

    arch["modules"] = sort_modules(arch["modules"])

    with open("outputs/subgraph_archs/subgraph_arch_merged.json", "w") as write_file:
        write_file.write(json.dumps(arch, indent=4, sort_keys=True))

    return arch


def construct_eq(in0, in1, op):

    op_str_map = {}

    op_str_map["mul"] = "(in_0 * in_1)"
    op_str_map["not"] = "(~in_0)"
    op_str_map["and"] = "(in_0 & in_1)"
    op_str_map["or"] = "(in_0 | in_1)"
    op_str_map["xor"] = "(in_0 ^ in_1)"
    op_str_map["shl"] = "(in_0 << in_1)"
    op_str_map["lshr"] = "(in_0 >> in_1)"
    op_str_map["ashr"] = "(in_0 >> in_1)"
    op_str_map["neg"] = "(-in_0)"
    op_str_map["add"] = "(in_0 + in_1)"
    op_str_map["sub"] = "(in_0 - in_1)"
    op_str_map["sle"] = "(Bit(1) if in_0 <= in_1 else Bit(0))"
    op_str_map["sge"] = "(Bit(1) if in_0 >= in_1 else Bit(0))"
    op_str_map["ule"] = "(Bit(1) if in_0 <= in_1 else Bit(0))"
    op_str_map["uge"] = "(Bit(1) if in_0 >= in_1 else Bit(0))"
    op_str_map["eq"] = "(Bit(1) if in_0 == in_1 else Bit(0))"
    op_str_map["slt"] = "(Bit(1) if in_0 < in_1 else Bit(0))"
    op_str_map["sgt"] = "(Bit(1) if in_0 > in_1 else Bit(0))"
    op_str_map["ult"] = "(Bit(1) if in_0 < in_1 else Bit(0))"
    op_str_map["ugt"] = "(Bit(1) if in_0 > in_1 else Bit(0))"

    return op_str_map[op].replace("in_0", in0).replace("in_1", in1)


def subgraph_to_peak(subgraph, sub_idx, b, op_types):
    node_dict = {}


    for n, d in subgraph.nodes.data(True):
        if op_types[subgraph.nodes[n]['alu_op']] != "input" \
            and op_types[subgraph.nodes[n]['alu_op']] != "const_input":

            pred = {}
            pred['alu_op'] = op_types[subgraph.nodes[n]['alu_op']]
            for s in subgraph.pred[n]:
                pred[subgraph.edges[(s, n, 0)]['port']] = b[s]
            node_dict[b[n]] = pred

    ret_val = node_dict.copy()
    args_str = ""
    eq_dict = {}
    inputs = set()

    while len(node_dict) > 0:
        for node, data in node_dict.copy().items():
            if data['alu_op'] == 'output':
                node_dict.pop(node)
            elif data['alu_op'] == 'const':
                eq_dict[node] = data['0']
                node_dict.pop(node)
                args_str += data['0'] + " : Const(Data), "
            else:
                if ("in" in data['0']
                        or data['0'] in eq_dict) and ("in" in data['1']
                                                        or data['1'] in eq_dict):
                    eq_dict[node] = construct_eq(
                        str(eq_dict.get(data['0'], data['0'])),
                        str(eq_dict.get(data['1'], data['1'])), data['alu_op'])
                    last_eq = eq_dict[node]
                    node_dict.pop(node)
                if "in" in data['1']:
                    inputs.add(data['1'])

            if "in" in data['0']:
                inputs.add(data['0'])

    print(last_eq)

    for i in inputs:
        args_str += i + " : Data, "

    args_str = args_str[:-2]

    peak_output = '''
from peak import Peak, family_closure, Const
from peak import family
from peak.family import AbstractFamily

@family_closure
def mapping_function_fc(family: AbstractFamily):
    Data = family.BitVector[16]
    Bit = family.Bit
    @family.assemble(locals(), globals())
    class mapping_function(Peak):
        def __call__(self, ''' + args_str + ''') -> Data:
  
            return ''' + last_eq + '''
      
    return mapping_function
'''

    if not os.path.exists('outputs/peak_eqs'):
        os.makedirs('outputs/peak_eqs')

    with open("outputs/peak_eqs/peak_eq_" + str(sub_idx) + ".py", "w") as write_file:
        write_file.write(peak_output)

    return ret_val


def gen_rewrite_rule(node_dict):
    rr = {}
    for k, v in node_dict.items():
        rr[k] = {}
        rr[k]['0'] = v['0']
        if not (v['alu_op'] == "const" or v['alu_op'] == "output"):
            rr[k]['1'] = v['1']
        rr[k]['alu_op'] = v['alu_op']

    # print(rr)
    return rr


def formulate_rewrite_rules(rrules, merged_arch):

    rrules_out = []

    arch = read_arch("./outputs/subgraph_archs/subgraph_arch_merged.json")
    graph_arch(arch)
    PE_fc = wrapped_pe_arch_closure(arch)
    arch_mapper = ArchMapper(PE_fc)


    for sub_idx, rrule in enumerate(rrules):
        rr_output = {}
      
        input_mappings = {}
        const_mappings = {}
        seen_inputs = []

        alu = []
        mul = []
        mux_in0 = []
        mux_in1 = []
        mux_out = []
        cfg_idx = 0

        for module in merged_arch["modules"]:
            if module["id"] in rrule:
                v = rrule[module["id"]]

                if module["type"] == "alu" or module["type"] == "mul":
                    if module["type"] == "alu":
                        alu.append(v['alu_op'])
                    elif module["type"] == "mul":
                        mul.append("Mult0")

                    if len(module["in0"]) > 1:
                        for mux_idx, mux_in in enumerate(module["in0"]):
                            if v['0'] == mux_in:
                                mux_in0.append(mux_idx)

                                if "in" in v['0']:
                                    in_idx = int(mux_in.split("in")[1])
                                    input_mappings[in_idx] = v['0']
                                    seen_inputs.append(v['0'])
                            elif "in" in mux_in:
                                in_idx = int(mux_in.split("in")[1])

                                if mux_in not in seen_inputs:
                                    input_mappings[in_idx] = 0
                                    seen_inputs.append(mux_in)

                    elif "in" in module["in0"][0]:
                        in_idx = int(v['0'].split("in")[1])
                        input_mappings[in_idx] = v['0']
                        seen_inputs.append(v['0'])

                    if len(module["in1"]) > 1:
                        for mux_idx, mux_in in enumerate(module["in1"]):
                            if v['1'] == mux_in:
                                mux_in1.append(mux_idx)

                                if "in" in v['1']:
                                    in_idx = int(mux_in.split("in")[1])
                                    input_mappings[in_idx] = v['1']
                                    seen_inputs.append(v['1'])
                            elif "in" in mux_in:
                                in_idx = int(mux_in.split("in")[1])

                                if mux_in not in seen_inputs:
                                    input_mappings[in_idx] = 0
                                    seen_inputs.append(mux_in)

                    elif "in" in module["in1"][0]:
                        in_idx = int(v['1'].split("in")[1])
                        input_mappings[in_idx] = v['1']
                        seen_inputs.append(v['1'])

                elif module["type"] == "const":
                    # cfg_idx = int(v['0'].split("const")[1])
                    const_mappings[cfg_idx] = v['0']
                    cfg_idx += 1


            else:
                if module["type"] == "alu" or module["type"] == "mul":
                    if module["type"] == "alu":
                        alu.append("add")
                    elif module["type"] == "mul":
                        mul.append("Mult0") 

                    if len(module["in0"]) > 1:
                        mux_in0.append(0)

                    if len(module["in1"]) > 1:
                        mux_in1.append(0)

                    for mux_in in module["in0"]:
                        if "in" in mux_in:
                            in_idx = int(mux_in.split("in")[1])

                            if mux_in not in seen_inputs:
                                input_mappings[in_idx] = 0
                                seen_inputs.append(mux_in)

                    for mux_in in module["in1"]:
                        if "in" in mux_in:
                            in_idx = int(mux_in.split("in")[1])

                            if mux_in not in seen_inputs:
                                input_mappings[in_idx] = 0
                                seen_inputs.append(mux_in)

                elif module["type"] == "const":
                    const_mappings[cfg_idx] = 0
                    cfg_idx += 1
            

        
        output_id = rrule["out0"]["0"]
        if len(arch.outputs[0]) > 1:
            mux_out.append(arch.outputs[0].index(output_id))

        input_binding = []

        for k, v in input_mappings.items():
            if v == 0:
                input_binding.append((peak.mapper.utils.Unbound, ("inputs", k)))
            else:
                input_binding.append(((v,), ("inputs", k)))

        for k, v in const_mappings.items():
            if v == 0:
                input_binding.append((peak.mapper.utils.Unbound, ("inst", "const_data", k)))
            else:
                input_binding.append(((v,), ("inst", "const_data", k)))


        op_map = {}

        op_map["Mult0"] = MUL_t.Mult0
        op_map["Mult1"] = MUL_t.Mult1
        op_map["not"] = ALU_t.Sub  
        op_map["and"] = ALU_t.And    
        op_map["or"] = ALU_t.Or    
        op_map["xor"] = ALU_t.XOr    
        op_map["shl"] = ALU_t.SHL   
        op_map["lshr"] = ALU_t.SHR    
        op_map["ashr"] = ALU_t.SHR    
        op_map["neg"] = ALU_t.Sub   
        op_map["add"] = ALU_t.Add    
        op_map["sub"] = ALU_t.Sub    
        op_map["sle"] = ALU_t.LTE_Min    
        op_map["sge"] = ALU_t.GTE_Max     
        op_map["ule"] = ALU_t.LTE_Min   
        op_map["uge"] = ALU_t.GTE_Max    
        op_map["eq"] = ALU_t.Sub    
        op_map["slt"] = ALU_t.LTE_Min   
        op_map["sgt"] = ALU_t.GTE_Max  
        op_map["ult"] = ALU_t.LTE_Min    
        op_map["ugt"] = ALU_t.GTE_Max    

        alu = [op_map[n] for n in alu]
        mul = [op_map[n] for n in mul]

        mux_in0_bw = [m.math.log2_ceil(len(arch.modules[i].in0)) for i in range(len(arch.modules)) if len(arch.modules[i].in0) > 1]
        mux_in1_bw = [m.math.log2_ceil(len(arch.modules[i].in1)) for i in range(len(arch.modules)) if len(arch.modules[i].in1) > 1]

        breakpoint()

        mux_in0_asmd = [BitVector[mux_in0_bw[i]](n) for i, n in enumerate(mux_in0)]
        mux_in1_asmd = [BitVector[mux_in1_bw[i]](n) for i, n in enumerate(mux_in1)]

        mux_out_bw = [m.math.log2_ceil(len(arch.outputs[i])) for i in range(arch.num_outputs) if len(arch.outputs[i]) > 1]
        mux_out_asmd = [BitVector[mux_out_bw[i]](n) for i, n in enumerate(mux_out)]


        # inst_gen = gen_inst(alu=alu, mul=mul, mux_in0=mux_in0, mux_in1=mux_in1, mux_out=mux_out)
        # input_binding.append((assembler(inst_gen), ("inst",)))

        # constraint = {}

        for ind, a in enumerate(alu):
            input_binding.append((a, ("inst", "alu", ind)))
            input_binding.append((BitVector[5](0), ("inst", "cond", ind)))

        for ind, a in enumerate(mul):
            input_binding.append((a, ("inst", "mul", ind)))

        for ind, a in enumerate(mux_in0_asmd):
            input_binding.append((a, ("inst", "mux_in0", ind)))

        for ind, a in enumerate(mux_in1_asmd):
            input_binding.append((a, ("inst", "mux_in1", ind)))

        for ind, a in enumerate(mux_out_asmd):
            input_binding.append((a, ("inst", "mux_out", ind)))

        input_binding.append((BitVector[1](0), ('inst', 'signed')))
        input_binding.append((BitVector[8](0), ('inst', 'lut')))



        constrained_vars = {"inst", "inputs"}

        for i in arch_mapper.input_varmap:
            if i[0] not in constrained_vars:
                input_binding.append((peak.mapper.utils.Unbound, i))

        new_rrule = {}

        # constraints.append(constraint)
        new_rrule["ibinding"] = input_binding

        new_rrule["obinding"] = [((0,), ('pe_outputs', 0))]

        rrules_out.append(new_rrule)

    return rrules_out


def test_rewrite_rules(rrules):
    arch = read_arch("./outputs/subgraph_archs/subgraph_arch_merged.json")
    PE_fc = wrapped_pe_arch_closure(arch)
    Inst_fc = inst_arch_closure(arch)
    Inst = Inst_fc(family.PyFamily())
 

    inst_type = PE_fc(family.PyFamily()).input_t.field_dict["inst"]

    _assembler = Assembler(inst_type)
    assembler = _assembler.assemble

    asm_fc = asm_arch_closure(arch)
    gen_inst = asm_fc(family.PyFamily())


    for rr_ind, rrule in enumerate(rrules):

        print("Rewrite rule ", rr_ind)
        # for c in rrule["ibinding"]:
        #     print(c)
        # for c in rrule["obinding"]:
        #     print(c)

        arch_mapper = ArchMapper(PE_fc)

        peak_eq = importlib.import_module("outputs.peak_eqs.peak_eq_" + str(rr_ind))


        # ir_mapper = arch_mapper.process_ir_instruction(peak_eq.mapping_function_fc)

        # solution = ir_mapper.solve(external_loop=True)

        rr = RewriteRule(rrule["ibinding"], rrule["obinding"], peak_eq.mapping_function_fc, PE_fc)

        counter_example = rr.verify()
        
        if counter_example is not None: 
            print(counter_example)
            exit()
        else:
            print("PASSED rewrite rule verify")

        rrules[rr_ind] = rr
    
    return rrules



def write_rewrite_rules(rrules):
    for sub_idx, rrule in enumerate(rrules):
        rrule_out = {}
        rrule_out["ibinding"] = []
        for t in rrule.ibinding:
            if isinstance(t[0], BitVector):
                rrule_out["ibinding"].append(tuple([{'type':'BitVector', 'width':len(t[0]), 'value':t[0].value}, t[1]]))
            elif isinstance(t[0], Bit):
                rrule_out["ibinding"].append(tuple([{'type':'Bit', 'width':1, 'value':t[0]._value}, t[1]]))
            elif t[0] == peak.mapper.utils.Unbound:
                rrule_out["ibinding"].append(tuple(["unbound", t[1]]))
            else:
                rrule_out["ibinding"].append(t)

        rrule_out["obinding"] = []
        for t in rrule.obinding:
            if t[0] == peak.mapper.utils.Unbound:
                rrule_out["obinding"].append(tuple(["unbound", t[1]]))
            else:
                rrule_out["obinding"].append(t)

        if not os.path.exists('outputs/subgraph_rewrite_rules'):
            os.makedirs('outputs/subgraph_rewrite_rules')

        with open("outputs/subgraph_rewrite_rules/subgraph_rr_" + str(sub_idx) + ".json", "w") as write_file:
            write_file.write(json.dumps(rrule_out))

def clean_output_dirs():
    if os.path.exists('outputs/subgraph_rewrite_rules'):
        shutil.rmtree("outputs/subgraph_rewrite_rules") 
    if os.path.exists('outputs/subgraph_archs'):
        shutil.rmtree("outputs/subgraph_archs")
    if os.path.exists('outputs/peak_eqs'):
        shutil.rmtree("outputs/peak_eqs")  

def merge_subgraphs(file_ind_pairs):
    clean_output_dirs()

    with open(".temp/op_types.txt") as file:
        op_types_flipped = ast.literal_eval(file.read())

    op_types = {str(v): k for k, v in op_types_flipped.items()}
    op_types["-1"] = "alu"
    op_types["input"] = "input"
    op_types["const_input"] = "const_input"
    op_types["output"] = "output"

    alu_supported_ops = [
        "not", "and", "or", "xor", "shl", "lshr", "ashr", "neg", "add", "sub",
        "sle", "sge", "ule", "uge", "eq", "mux", "slt", "sgt", "ult", "ugt"
    ]

    graphs = []

    for subgraph_file, inds in file_ind_pairs.items():
        with open(subgraph_file) as file:
            lines = file.readlines()

        graph_ind = -1
        graphs_per_file = []

        for line in lines:
            if ':' in line:
                graph_ind += 1
                graphs_per_file.append(nx.MultiDiGraph())
            elif 'v' in line:
                if op_types[line.split()[2]] in alu_supported_ops:
                    op = "-1"
                else:
                    op = line.split()[2]
                graphs_per_file[graph_ind].add_node(
                    line.split()[1], op=op, alu_op=line.split()[2])
            elif 'e' in line:
                graphs_per_file[graph_ind].add_edge(
                    line.split()[1], line.split()[2], port=line.split()[3])

        graphs += [graphs_per_file[int(i)] for i in inds]

    graphs_no_input_nodes = copy.deepcopy(graphs)

    for graph in graphs:
        add_input_and_output_nodes(graph, op_types)

    rrules = []

    G = graphs[0]
    mappings = []

    for i in range(1, len(graphs)):
        gc, g1_map, g2_map = construct_compatibility_graph(G, graphs[i], op_types, op_types_flipped)

        C = FindMaximumWeightClique(gc)

        print("subgraph ", i)
        G, new_mapping = reconsruct_resulting_graph(C, G, graphs[i], g1_map, g2_map, op_types)

        if i == 1:
            print("subgraph ", 0)
            new_mapping_0 = {k:k for k in graphs[0].nodes}
            new_mapping_0.update({u + ", " + v:u + ", " + v for (u,v,z) in graphs[0].edges})
            node_dict = subgraph_to_peak(graphs[0], 0, new_mapping_0, op_types)
            rrules.append(
                gen_rewrite_rule(node_dict))
            mappings.append(new_mapping_0)
            print()

        mappings.append(new_mapping)
        node_dict = subgraph_to_peak(graphs[i], i, new_mapping, op_types)
        rrules.append(gen_rewrite_rule(node_dict))

        print()


    merged_arch = merged_subgraph_to_arch(G, op_types)
    constraints = formulate_rewrite_rules(rrules, merged_arch)

    tic = time.perf_counter()
    rrules = test_rewrite_rules(constraints)
    toc = time.perf_counter()
    print(f"{toc - tic:0.4f} seconds")

    # while rr_test_res != -1:

    #     resolve_rr_gen_fail(rr_test_res, mappings, graphs, G)
    #     merged_arch = merged_subgraph_to_arch(G, op_types)
    #     complete_rr = formulate_rewrite_rules(rrules, merged_arch)
    #     rr_test_res = test_rewrite_rules(complete_rr)

    write_rewrite_rules(rrules)
DEBUG = False