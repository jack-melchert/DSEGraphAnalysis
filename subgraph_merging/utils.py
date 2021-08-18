import sys
import os
import json
import copy
import shutil 
import ast
import pickle
import networkx as nx
import magma as m

from peak_gen.sim import pe_arch_closure
from peak_gen.arch import read_arch, graph_arch

from peak import family

import subgraph_merging.config as config
from .subgraph import Subgraph, DSESubgraph


def read_subgraphs(file_ind_pairs):
    graphs = []
    graphs_sizes = {}
    name_mappings = []

    for subgraph_file, inds in file_ind_pairs.items():
        with open(subgraph_file) as file:
            lines = file.readlines()

        graph_ind = -1
        graphs_per_file = []
        for line in lines:
            if ':' in line:
                graph_ind += 1
                graphs_per_file.append(nx.MultiDiGraph())
                graphs_sizes[graph_ind] = 0
                name_mapping = {}
                name_mappings.append(name_mapping)
            elif 'v' in line:
                op_in = config.op_types[line.split()[2]]
                if op_in != "none":
                    if op_in == "and" or  op_in == "or" or op_in == "xor":
                        op = 'bit_alu'
                    elif op_in == "smax" or op_in == "umax" or op_in == "sge" or op_in == "uge":
                        op = "gte"
                    elif op_in == "smin" or op_in == "umin" or op_in == "sle" or op_in == "ule":
                        op = "lte"
                    elif op_in == "slt" or op_in == "sgt" or op_in == "ult" or op_in == "ugt" or op_in == "eq":
                        op = "sub"
                    elif op_in == "ashr" or op_in == "lshr":
                        op = "shr"
                    elif op_in == "mult_middle":
                        op = "mul"
                    elif op_in in config.lut_supported_ops:
                        op = "lut"
                    else:
                        op = line.split()[2]
                    if op in config.op_types_flipped:
                        op = config.op_types_flipped[op]
                    graphs_per_file[graph_ind].add_node(
                        line.split()[1], op=op, op_config=[line.split()[2]])
                    graphs_sizes[graph_ind] += config.weights[op_in]
                    name_mappings[graph_ind][line.split()[1]] = str(config.node_counter)
                    config.node_counter += 1
 
            elif 'e' in line:
                graphs_per_file[graph_ind].add_edge(
                    line.split()[1], line.split()[2], port=line.split()[3], regs=0)

        relabeled_graphs = []
        for ind, graph in enumerate(graphs_per_file):
            relabeled_graphs.append(nx.relabel_nodes(graph, name_mappings[ind]))

        sorted_graphs = sorted(graphs_sizes.items(), key=lambda x: x[1], reverse=True)
        graphs += [relabeled_graphs[i[0]] for i in sorted_graphs if i[0] in inds]

    subgraphs = []
    for ind, graph in enumerate(graphs):
        subgraphs.append(DSESubgraph(graph))
    return subgraphs

def is_node_input_or_output(node):

    input_output_names = config.input_names.union(config.output_names)

    if 'op' in node:
        return node['op'] in input_output_names
    elif 'op0' in node and 'op1' in node:
        return node['op0'] in input_output_names or node['op1'] in input_output_names
    else:
        raise ValueError


def is_node_output(node):

    if 'op' in node:
        return node['op'] in config.output_names
    elif 'op0' in node and 'op1' in node:
        return node['op0'] in config.output_names or node['op1'] in config.output_names
    else:
        raise ValueError

def is_node_input_or_const(node):

    if 'op' in node:
        return node['op'] in config.input_names | config.const_names
    elif 'op0' in node and 'op1' in node:
        return node['op0'] in config.input_names | config.const_names or node['op1'] in config.input_names | config.const_names
    else:
        raise ValueError

def is_node_input(node):

    if 'op' in node:
        return node['op'] in config.input_names
    elif 'op0' in node and 'op1' in node:
        return node['op0'] in config.input_names or node['op1'] in config.input_names
    else:
        raise ValueError

def is_node_bit_input(node):

    if 'op' in node:
        return node['op'] in config.input_names and "bit" in node['op']
    elif 'op0' in node and 'op1' in node:
        return (node['op0'] in config.input_names and "bit" in node['op0']) or (node['op1'] in config.input_names and "bit" in node['op1'])
    else:
        raise ValueError

def add_primitive_ops(subgraphs):

    with open(".temp/used_ops.txt", "rb") as file:
        used_ops = pickle.load(file)

    print(used_ops)

    graph_weights = {}
    primitive_graphs = {}

    for ind, op in enumerate(used_ops):
        op_graph = nx.MultiDiGraph()
        
        if op == "and" or  op == "or" or op == "xor":
            op_t = 'bit_alu'
            op_graph.add_node(str(config.node_counter), op=config.op_types_flipped[op_t], op_config=[config.op_types_flipped[op]])
        elif op == "smax" or op == "umax" or op == "sge" or op == "uge":
            op_t = "gte"
            op_graph.add_node(str(config.node_counter), op=config.op_types_flipped[op_t], op_config=[config.op_types_flipped[op]])
        elif op == "smin" or op == "umin" or op == "sle" or op == "ule":
            op_t = "lte"
            op_graph.add_node(str(config.node_counter), op=config.op_types_flipped[op_t], op_config=[config.op_types_flipped[op]])
        elif op == "slt" or op == "sgt" or op == "ult" or op == "ugt" or op == "eq":
            op_t = "sub"
            op_graph.add_node(str(config.node_counter), op=config.op_types_flipped[op_t], op_config=[config.op_types_flipped[op]])
        elif op == "ashr" or op == "lshr":
            op_t = "shr"
            op_graph.add_node(str(config.node_counter), op=config.op_types_flipped[op_t], op_config=[config.op_types_flipped[op]])
        elif op == "mult_middle":
            op_t = "mul"
            op_graph.add_node(str(config.node_counter), op=config.op_types_flipped[op_t], op_config=[config.op_types_flipped[op]])
        elif op in config.lut_supported_ops:
            op_t = "lut"
            op_graph.add_node(str(config.node_counter), op=config.op_types_flipped[op_t], op_config=[config.op_types_flipped[op]])
        else:
            op_graph.add_node(str(config.node_counter), op=config.op_types_flipped[op], op_config=[config.op_types_flipped[op]])
        config.node_counter += 1        
      
        graph_weights[ind] = config.weights[op]
        primitive_graphs[ind] = op_graph

    sorted_graphs = sorted(graph_weights.items(), key=lambda x: x[1], reverse=True)
    for i in sorted_graphs:
        subgraphs.append(DSESubgraph(primitive_graphs[i[0]]))




def clean_output_dirs():
    if os.path.exists('outputs'):
        shutil.rmtree("outputs") 
    os.makedirs('outputs')

    
def sort_modules(modules, subgraph):
    ids = []
    output_modules = []
    sort_count = 1000
    modules_save = modules.copy()
    while len(modules) > 0:
        for module in modules.copy():
            if module['type'] == 'const' or module['type'] == 'bitconst':
                ids.append(module["id"])
                output_modules.append(module)
                modules.remove(module)
                sort_count = 1000
            elif module['type'] == 'reg' or module['type'] == 'bitreg':
                inorder = True

                for in0_item in module["in"]:
                    if "reg" in in0_item:
                        inorder = inorder and in0_item in ids
                    else:
                        inorder = inorder and (is_node_input_or_const(subgraph.nodes(data = True)[in0_item]) or in0_item in ids)

                if inorder:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)
                    sort_count = 1000
            else:
            
                inorder = True

                for in0_item in module["in0"]:
                    if "reg" in in0_item:
                        inorder = inorder and in0_item in ids
                    else:
                        inorder = inorder and (is_node_input_or_const(subgraph.nodes(data = True)[in0_item]) or in0_item in ids)

                for in1_item in module["in1"]:
                    if "reg" in in1_item:
                        inorder = inorder and in1_item in ids
                    else:
                        inorder = inorder and (is_node_input_or_const(subgraph.nodes(data = True)[in1_item]) or in1_item in ids)

                if module['type'] == 'mux' or module['type'] == 'bitmux':
                    for sel_item in module["in2"]:
                        if "reg" in sel_item:
                            inorder = inorder and sel_item in ids
                        else:
                            inorder = inorder and (is_node_input_or_const(subgraph.nodes(data = True)[sel_item]) or sel_item in ids)

                if inorder:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)
                    sort_count = 1000

        sort_count -= 1
        if sort_count == 0:
            print("There is a cycle in the merged graph!")
            for i in modules_save:
                print(i)
    return output_modules


def construct_eq(in0, in1, op, absd_count, in2=""):

    op_str_map = {}

    op_str_map["mul"] = "Data(UInt(in_0) * UInt(in_1))"
    op_str_map["mult_middle"] = "(Data32(in_0) * Data32(in_1))[8:24]"
    op_str_map["not"] = "Data(~UInt(in_0))"
    op_str_map["and"] = "Data(UInt(in_0) & UInt(in_1))"
    op_str_map["or"] = "Data(UInt(in_0) | UInt(in_1))"
    op_str_map["xor"] = "Data(UInt(in_0) ^ UInt(in_1))"
    op_str_map["shl"] = "Data(UInt(in_0) << UInt(in_1))"
    op_str_map["lshr"] = "Data(UInt(in_0) >> UInt(in_1))"
    op_str_map["ashr"] = "Data(SInt(in_0) >> SInt(in_1))"
    op_str_map["neg"] = "Data(-UInt(in_0))"
    op_str_map["add"] = "Data(UInt(in_0) + UInt(in_1))"
    op_str_map["sub"] = "Data(UInt(in_0) - UInt(in_1))"
    op_str_map["sle"] = "Bit(SInt(in_0) <= SInt(in_1))"
    op_str_map["sge"] = "Bit(SInt(in_0) >= SInt(in_1))"
    op_str_map["ule"] = "Bit(UInt(in_0) <= UInt(in_1))"
    op_str_map["uge"] = "Bit(UInt(in_0) >= UInt(in_1))"
    op_str_map["eq"] = "Bit(UInt(in_0) == UInt(in_1))"
    op_str_map["slt"] = "Bit(SInt(in_0) < SInt(in_1))"
    op_str_map["sgt"] = "Bit(SInt(in_0) > SInt(in_1))"
    op_str_map["ult"] = "Bit(UInt(in_0) < UInt(in_1))"
    op_str_map["ugt"] = "Bit(UInt(in_0) > UInt(in_1))"
    op_str_map["mux"] = "Data(in_2.ite(UInt(in_1),UInt(in_0)))"
    op_str_map["umax"] = "Data((UInt(in_0) >= UInt(in_1)).ite(UInt(in_0), UInt(in_1)))"
    op_str_map["umin"] = "Data((UInt(in_0) <= UInt(in_1)).ite(UInt(in_0), UInt(in_1)))"
    op_str_map["smax"] = "Data((SInt(in_0) >= SInt(in_1)).ite(SInt(in_0), SInt(in_1)))"
    op_str_map["smin"] = "Data((SInt(in_0) <= SInt(in_1)).ite(SInt(in_0), SInt(in_1)))"
    op_str_map["abs"] = "Data((SInt(in_0) >= SInt(S)).ite(SInt(in_0), SInt(in_0)*SInt(S1)))"
    op_str_map["bitand"] = "Bit(Bit(in_0) & Bit(in_1))"
    op_str_map["bitnot"] = "Bit(~in0)" 
    op_str_map["bitor"] = "Bit(Bit(in_0) | Bit(in_1))"
    op_str_map["bitxor"] = "Bit(Bit(in_0) ^ Bit(in_1))"
    op_str_map["bitmux"] = "Bit(in_2.ite(Bit(in_1), Bit(in_0)))"
    op_str_map["floatmul"] = "Data(UInt(in_0) * UInt(in_1))"
    op_str_map["floatadd"] = "Data(UInt(in_0) + UInt(in_1))"
    op_str_map["floatsub"] = "Data(UInt(in_0) - UInt(in_1))"

    sub = "sub" + str(absd_count)

    op_str_map["absd"] = "Data((" + sub + " >= SInt(0)).ite(" + sub + ", (SInt(-1)*" + sub + ")))"

    if op == "absd":
        absd_str = sub + " = SInt(in_0 - in_1); "
        absd_count += 1
    else:
        absd_str = ""

    return op_str_map[op].replace("in_0", in0).replace("in_1", in1).replace(
        "in_2", in2), absd_str.replace("in_0", in0).replace("in_1", in1), absd_count



def swap_ports(subgraph, dest_node):
    for s in subgraph.pred[dest_node]:
        if subgraph.edges[(s, dest_node, 0)]['port'] == "0":
            subgraph.edges[(s, dest_node, 0)]['port'] = "1"
        else:
            subgraph.edges[(s, dest_node, 0)]['port'] = "0"

def check_no_cycles(pair, g1, g2):

    a0 = pair[0][1]['start']
    a1 = pair[1][1]['start']

    b0 = pair[0][1]['end']
    b1 = pair[1][1]['end']


    apath = nx.algorithms.shortest_paths.generic.has_path(g1, a0, a1)
    apath_r = nx.algorithms.shortest_paths.generic.has_path(g1, a1, a0)
    bpath = nx.algorithms.shortest_paths.generic.has_path(g2, b0, b1)
    bpath_r = nx.algorithms.shortest_paths.generic.has_path(g2, b1, b0)

    cycle_exists = (apath and bpath_r) or (apath_r and bpath)

    return not cycle_exists
    
def get_node_timing(g, v):
    if v == 'start' or v == 'end':
        return 0
    else:
        if g.nodes(data = True)[v]['op'] in config.op_types and config.op_types[g.nodes(data = True)[v]['op']] in config.op_map:
            return config.op_costs[config.op_map[config.op_types[g.nodes(data = True)[v]['op']]]]["crit_path"]
        else:
            # print(f"Couldn't find node {config.op_types[g.nodes(data = True)[v]['op']]} in op_costs")
            return 0.01


def predecessors(g, v):
    ret = list(g.predecessors(v))
    return ret

def successors(g, v):
    ret = list(g.successors(v))
    return ret

def get_clock_period(g):
    critical_path = 0

    for path in nx.all_simple_paths(g, source='start', target='end'):
        temp_crit_path = 0

        for u, v in zip(path, path[1:]):
            temp_crit_path += get_node_timing(g, u)

            if g[u][v].get(0)['regs'] > 0:
                critical_path = max(critical_path, temp_crit_path)
                temp_crit_path = 0

        critical_path = max(critical_path, temp_crit_path)

    return critical_path

def gen_verilog():
    arch = read_arch("outputs/PE.json")
    graph_arch(arch)
    PE_fc = pe_arch_closure(arch)
    PE = PE_fc(family.MagmaFamily())

    if not os.path.exists('outputs/verilog'):
        os.makedirs('outputs/verilog')
    m.compile(f"outputs/verilog/PE", PE, output="coreir-verilog")

def print_green(string):
    CGREEN  = '\33[32m'
    CEND = '\033[0m'
    print(CGREEN + string + CEND)

def print_red(string):
    CRED = '\033[91m'
    CEND = '\033[0m'
    print(CRED + string + CEND)