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
                    elif op_in in config.lut_supported_ops:
                        op = "lut"
                    else:
                        op = line.split()[2]
                    graphs_per_file[graph_ind].add_node(
                        line.split()[1], op=op, op_config=[line.split()[2]])
                    graphs_sizes[graph_ind] += config.weights[op_in]
                    name_mappings[graph_ind][line.split()[1]] = str(config.node_counter)
                    config.node_counter += 1
 
            elif 'e' in line:
                graphs_per_file[graph_ind].add_edge(
                    line.split()[1], line.split()[2], port=line.split()[3])

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

    input_output_names = config.input_names.union(config.input_names)

    if 'op' in node:
        return node['op'] in input_output_names
    elif 'op0' in node and 'op1' in node:
        return node['op0'] in input_output_names or node['op1'] in input_output_names
    else:
        raise ValueError

def is_node_input(node):

    if 'op' in node:
        return node['op'] in config.input_names
    elif 'op0' in node and 'op1' in node:
        return node['op0'] in config.input_names or node['op1'] in config.input_names
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
            else:
            
                inorder = True

                for in0_item in module["in0"]:
                    inorder = inorder and (is_node_input(subgraph.nodes(data = True)[in0_item]) or in0_item in ids)

                for in1_item in module["in1"]:
                    inorder = inorder and (is_node_input(subgraph.nodes(data = True)[in1_item]) or in1_item in ids)

                if module['type'] == 'mux' or module['type'] == 'bitmux':
                    for sel_item in module["in2"]:
                        inorder = inorder and (is_node_input(subgraph.nodes(data = True)[sel_item]) or sel_item in ids)

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


def merged_subgraph_to_arch(subgraph):

    arch = {}
    arch["input_width"] = 16
    arch["output_width"] = 16
    arch["enable_input_regs"] = False
    arch["enable_output_regs"] = False
    modules = {}
    ids = []
    connected_ids = []

    for n, d in subgraph.nodes.data(True):

        # Don't want to put input and output nodes in actual arch
        if d["op"] != "output" and d["op"] != "bit_output" and d["op"] != "const_input" and d["op"] != "bit_const_input":

            # Only want id of input nodes
            if d["op"] != "input" and d["op"] != "bit_input":
                modules[n] = {}
                modules[n]["id"] = n

                op = config.op_types[d["op"]]
                op_config = [config.op_types[x] for x in d["op_config"]]

                # if op == "lut":
                #     modules[n]["type"] = 'lut'
                # elif op in bitwise_ops:
                #     modules[n]["type"] = 'bit_alu'
                # elif op == "smax" or op == "umax" or op == "sge" or op == "uge":
                #     modules[n]["type"] = "gte"
                # elif op == "smin" or op == "umin" or op == "sle" or op == "ule":
                #     modules[n]["type"] = "lte"
                # elif op == "slt" or op == "sgt" or op == "ult" or op == "ugt" or op == "eq":
                #     modules[n]["type"] = "sub"
                # elif op == "ashr" or op == "lshr":
                #     modules[n]["type"] = "shr"
                # else:
                modules[n]["type"] = op
                    # for alu_op in op_config:
                    #     if alu_op not in alu_supported_ops and alu_op not in config.lut_supported_ops:
                    #         print("Warning: possible unsupported ALU operation found in subgraph:", n)
                    #     if alu_op in fp_alu_supported_ops:
                    #         op = "fp_alu"
                    # if op == "lut":
                    #     modules[n]["type"] = 'lut'
                    # elif op == "fp_alu":
                    #     modules[n]["type"] = 'fp_alu'
                    # else:
                    #     modules[n]["type"] = 'alu'

            ids.append(n)

    outputs = set()
    bit_outputs = set()

    for u, v, d in subgraph.edges.data(True):
        if v in modules:
            connected_ids.append(u)
            if modules[v]["type"] != "const" and modules[v]["type"] != "bitconst":
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
                elif d["port"] == "2":
                    if "in2" in modules[v]:
                        modules[v]["in2"].append(u)
                    else:
                        modules[v]["in2"] = [u]
        
        # Add to output mux
        if subgraph.nodes.data(True)[v]["op"] == "output":
            # op_config = [config.op_types[x] for x in subgraph.nodes.data(True)[u]["op_config"]]
            # for alu_op in op_config:
            #     if alu_op in config.bit_output_ops:
            #         bit_outputs.add(u)

            outputs.add(u)

        if subgraph.nodes.data(True)[v]["op"] == "bit_output":
            bit_outputs.add(u)

    arch["modules"] = [v for v in modules.values()]
    arch["outputs"] = [list(outputs)]
    if len(bit_outputs) > 0:
        arch["bit_outputs"] = [list(bit_outputs)]
    else:
        arch["bit_outputs"] = []

    if not os.path.exists('outputs/'):
        os.makedirs('outputs/')

    arch["modules"] = sort_modules(arch["modules"])

    with open("outputs/PE.json", "w") as write_file:
        write_file.write(json.dumps(arch, indent=4, sort_keys=True))

    return arch


def construct_eq(in0, in1, op, absd_count, in2=""):

    op_str_map = {}

    op_str_map["mul"] = "Data(in_0 * in_1)"
    op_str_map["not"] = "Data(~in_0)"
    op_str_map["and"] = "Data(in_0 & in_1)"
    op_str_map["or"] = "Data(in_0 | in_1)"
    op_str_map["xor"] = "Data(in_0 ^ in_1)"
    op_str_map["shl"] = "Data(in_0 << in_1)"
    op_str_map["lshr"] = "Data(in_0 >> in_1)"
    op_str_map["ashr"] = "Data(SData(in_0) >> SData(in_1))"
    op_str_map["neg"] = "Data(-in_0)"
    op_str_map["add"] = "Data(in_0 + in_1)"
    op_str_map["sub"] = "Data(in_0 - in_1)"
    op_str_map["sle"] = "Bit(SData(in_0) <= SData(in_1))"
    op_str_map["sge"] = "Bit(SData(in_0) >= SData(in_1))"
    op_str_map["ule"] = "Bit(in_0 <= in_1)"
    op_str_map["uge"] = "Bit(in_0 >= in_1)"
    op_str_map["eq"] = "Bit(in_0 == in_1)"
    op_str_map["slt"] = "Bit(SData(in_0) < SData(in_1))"
    op_str_map["sgt"] = "Bit(SData(in_0) > SData(in_1))"
    op_str_map["ult"] = "Bit(in_0 < in_1)"
    op_str_map["ugt"] = "Bit(in_0 > in_1)"
    op_str_map["mux"] = "Data(in_2.ite(in_1,in_0))"
    op_str_map["umax"] = "Data((in_0 >= in_1).ite(in_0, in_1))"
    op_str_map["umin"] = "Data((in_0 <= in_1).ite(in_0, in_1))"
    op_str_map["smax"] = "Data((SData(in_0) >= SData(in_1)).ite(SData(in_0), SData(in_1)))"
    op_str_map["smin"] = "Data((SData(in_0) <= SData(in_1)).ite(SData(in_0), SData(in_1)))"
    op_str_map["abs"] = "Data((SData(in_0) >= SData(0)).ite(SData(in_0), SData(in_0)*SData(-1)))"
    op_str_map["bitand"] = "Bit(in_0 & in_1)"
    op_str_map["bitnot"] = "Bit(~in0)" 
    op_str_map["bitor"] = "Bit(in_0 | in_1)"
    op_str_map["bitxor"] = "Bit(in_0 ^ in_1)"
    op_str_map["bitmux"] = "Bit(in_2.ite(in_1, in_0))"
    op_str_map["floatmul"] = "Data(in_0 * in_1)"
    op_str_map["floatadd"] = "Data(in_0 + in_1)"
    op_str_map["floatsub"] = "Data(in_0 - in_1)"

    sub = "sub" + str(absd_count)

    op_str_map["absd"] = "Data((" + sub + " >= SData(0)).ite(" + sub + ", (SData(-1)*" + sub + ")))"

    if op == "absd":
        absd_str = sub + " = SData(in_0 - in_1); "
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
    

def gen_verilog():
    arch = read_arch("outputs/PE.json")
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