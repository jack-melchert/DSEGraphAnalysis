import os
import json
import copy
import shutil 
import ast
import networkx as nx
import pickle
from peak_gen.sim import pe_arch_closure
from peak_gen.arch import read_arch, graph_arch
import magma as m
import sys
from peak import family

primitive_ops = {
    "and", "or", "xor", "shl", "lshr", "ashr", "add", "sub",
    "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", 
    "smax", "smin", "umax", "umin", "absd", "abs", "mul", "mux",
    "bitand", "bitor", "bitxor", "bitnot", "bitmux", "floatadd", "floatsub", "floatmul"
}

alu_supported_ops = {
    "and", "or", "xor", "shl", "lshr", "ashr", "add", "sub",
    "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", 
    "smax", "smin", "umax", "umin", "absd", "abs", "floatadd", "floatsub", "floatmul"
}

fp_alu_supported_ops = {
   "floatadd", "floatsub", "floatmul"
}

lut_supported_ops = {
    "bitand", "bitor", "bitxor", "bitnot", "bitmux"
}

bit_output_ops = {
    "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", "bitand", "bitor", "bitxor", "bitnot", "bitmux", "bitconst"
}


def read_subgraphs(file_ind_pairs, op_types):

    weights = {"const":1, "and":1, "or":1, "xor":1, "shl":1, "lshr":1, "ashr":1, "add":1, "sub":1,
    "sle":1, "sge":1, "ule":1, "uge":1, "eq":1, "slt":1, "sgt":1, "ult":1, "ugt":1, 
    "smax":2, "smin":2, "umax":2, "umin":2, "absd":4, "abs":2, "mul":1.5, "mux":1,
    "bitand":1, "bitor":1, "bitxor":1, "bitnot":1, "bitmux":1, "floatadd":1, "floatsub":1, "floatmul":1}

    graphs = []
    graphs_sizes = {}

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
            elif 'v' in line:
                op_in = op_types[line.split()[2]]
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
                    elif op_in in lut_supported_ops:
                        op = "lut"
                    else:
                        op = line.split()[2]
                    graphs_per_file[graph_ind].add_node(
                        line.split()[1], op=op, alu_ops=[line.split()[2]])
                    graphs_sizes[graph_ind] += weights[op_in]
            elif 'e' in line:
                graphs_per_file[graph_ind].add_edge(
                    line.split()[1], line.split()[2], port=line.split()[3])

        sorted_graphs = sorted(graphs_sizes.items(), key=lambda x: x[1], reverse=True)
        graphs += [graphs_per_file[i[0]] for i in sorted_graphs if i[0] in inds]

    return graphs

def read_optypes():
    with open(".temp/op_types.txt", "rb") as file:
        op_types_from_file = pickle.load(file)

    curr_ops = [*op_types_from_file]

    for op in primitive_ops:
        if op not in curr_ops:
            curr_ops.append(op)


    op_types = {str(k): v for k, v in enumerate(curr_ops)}

    special_ops = ["gte", "lte", "sub", "shr"]

    for op in special_ops:
        if op not in op_types:
            op_types[op] = op

    op_types["alu"] = "alu"
    op_types["bit_alu"] = "bit_alu"
    op_types["lut"] = "lut"
    op_types["input"] = "input"
    op_types["bit_input"] = "bit_input"
    op_types["const_input"] = "const_input"
    op_types["bit_const_input"] = "bit_const_input"
    op_types["output"] = "output"
    op_types["bit_output"] = "bit_output"

    op_types_flipped = {v: k for k, v in op_types.items()}

    return op_types, op_types_flipped

def add_primitive_ops(graphs, op_types_flipped):
    weights = {"const":1, "bitconst":1, "and":1, "or":1, "xor":1, "shl":1, "lshr":1, "ashr":1, "add":1, "sub":1,
    "sle":1, "sge":1, "ule":1, "uge":1, "eq":1, "slt":1, "sgt":1, "ult":1, "ugt":1, 
    "smax":2, "smin":2, "umax":2, "umin":2, "absd":4, "abs":2, "mul":1.5, "mux":1,
    "bitand":1, "bitor":1, "bitxor":1, "bitnot":1, "bitmux":1, "floatadd":1, "floatsub":1, "floatmul":1, "bit_alu":1,
    "gte":1, "lte":1, "sub":1, "shr":1}

    with open(".temp/used_ops.txt", "rb") as file:
        used_ops = pickle.load(file)

    print(used_ops)

    graph_weights = {}
    primitive_graphs = {}

    for ind, op in enumerate(used_ops):
        # if op != "const" and op != "bitconst":
        op_graph = nx.MultiDiGraph()
        
        if op == "and" or  op == "or" or op == "xor":
            op_t = 'bit_alu'
            op_graph.add_node('0', op=op_types_flipped[op_t], alu_ops=[op_types_flipped[op]])
        elif op == "smax" or op == "umax" or op == "sge" or op == "uge":
            op_t = "gte"
            op_graph.add_node('0', op=op_types_flipped[op_t], alu_ops=[op_types_flipped[op]])
        elif op == "smin" or op == "umin" or op == "sle" or op == "ule":
            op_t = "lte"
            op_graph.add_node('0', op=op_types_flipped[op_t], alu_ops=[op_types_flipped[op]])
        elif op == "slt" or op == "sgt" or op == "ult" or op == "ugt" or op == "eq":
            op_t = "sub"
            op_graph.add_node('0', op=op_types_flipped[op_t], alu_ops=[op_types_flipped[op]])
        elif op == "ashr" or op == "lshr":
            op_t = "shr"
            op_graph.add_node('0', op=op_types_flipped[op_t], alu_ops=[op_types_flipped[op]])
        elif op in lut_supported_ops:
            op_t = "lut"
            op_graph.add_node('0', op=op_types_flipped[op_t], alu_ops=[op_types_flipped[op]])
        else:
            op_graph.add_node('0', op=op_types_flipped[op], alu_ops=[op_types_flipped[op]])
        
      
        graph_weights[ind] = weights[op]
        primitive_graphs[ind] = op_graph

    sorted_graphs = sorted(graph_weights.items(), key=lambda x: x[1], reverse=True)
    for i in sorted_graphs:
        graphs.append(primitive_graphs[i[0]])




def clean_output_dirs():
    if os.path.exists('outputs/subgraph_rewrite_rules'):
        shutil.rmtree("outputs/subgraph_rewrite_rules") 
    if os.path.exists('outputs/subgraph_archs'):
        shutil.rmtree("outputs/subgraph_archs")
    if os.path.exists('outputs/peak_eqs'):
        shutil.rmtree("outputs/peak_eqs")  

def sort_modules(modules):
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
                    inorder = inorder and ("in" in in0_item or in0_item in ids)

                for in1_item in module["in1"]:
                    inorder = inorder and ("in" in in1_item or in1_item in ids)

                if module['type'] == 'mux' or module['type'] == 'bitmux':
                    for sel_item in module["in2"]:
                        inorder = inorder and ("in" in sel_item or sel_item in ids)

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


def merged_subgraph_to_arch(subgraph, op_types):



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

                op = op_types[d["op"]]
                alu_ops = [op_types[x] for x in d["alu_ops"]]

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
                    # for alu_op in alu_ops:
                    #     if alu_op not in alu_supported_ops and alu_op not in lut_supported_ops:
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
            # alu_ops = [op_types[x] for x in subgraph.nodes.data(True)[u]["alu_ops"]]
            # for alu_op in alu_ops:
            #     if alu_op in bit_output_ops:
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

    if not os.path.exists('outputs/subgraph_archs'):
        os.makedirs('outputs/subgraph_archs')

    arch["modules"] = sort_modules(arch["modules"])

    with open("outputs/subgraph_archs/subgraph_arch_merged.json", "w") as write_file:
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


def subgraph_to_peak(subgraph, sub_idx, b, op_types):
    absd_count = 0
    node_dict = {}


    for n, d in subgraph.nodes.data(True):
        if op_types[subgraph.nodes[n]['alu_ops'][0]] != "input" \
            and op_types[subgraph.nodes[n]['alu_ops'][0]] != "bit_input" \
            and op_types[subgraph.nodes[n]['alu_ops'][0]] != "const_input"\
            and op_types[subgraph.nodes[n]['alu_ops'][0]] != "bit_const_input":

            pred = {}
            pred['alu_op'] = op_types[subgraph.nodes[n]['alu_ops'][0]]
            for s in subgraph.pred[n]:
                pred[subgraph.edges[(s, n, 0)]['port']] = b[s]
            node_dict[b[n]] = pred

    ret_val = node_dict.copy()
    args_str = ""
    eq_dict = {}
    inputs = set()
    bit_inputs = set()
    absd_str = ""
    last_eq = ""
    output_type = ""

    while len(node_dict) > 0:
        for node, data in node_dict.copy().items():
            if data['alu_op'] == 'output' or data['alu_op'] == 'bit_output':
                node_dict.pop(node)
            elif data['alu_op'] == 'const':
                eq_dict[node] = data['0']
                node_dict.pop(node)
                args_str += data['0'] + " : Const(Data), "
            elif data['alu_op'] == 'bitconst':
                eq_dict[node] = data['0']
                node_dict.pop(node)
                args_str += data['0'] + " : Const(Bit), "
            else:
                if ("in" in data['0'] or data['0'] in eq_dict) and ("in" in data['1'] or data['1'] in eq_dict):

                    if '2' in data:
                        if ("in" in data['2'] or data['2'] in eq_dict):
                            eq_dict[node], absd_str_tmp, absd_count = construct_eq(
                            str(eq_dict.get(data['0'], data['0'])),
                            str(eq_dict.get(data['1'], data['1'])), data['alu_op'], absd_count,
                            str(eq_dict.get(data['2'], data['2'])))
                            last_eq = eq_dict[node]
                            output_type = "Bit" if data['alu_op'] in bit_output_ops else "Data"
                            node_dict.pop(node)
                            absd_str += absd_str_tmp
                    else:
                        eq_dict[node], absd_str_tmp, absd_count = construct_eq(
                            str(eq_dict.get(data['0'], data['0'])),
                            str(eq_dict.get(data['1'], data['1'])), data['alu_op'], absd_count)

                        absd_str += absd_str_tmp
                        last_eq = eq_dict[node]
                        output_type = "Bit" if data['alu_op'] in bit_output_ops else "Data"
                        node_dict.pop(node)
                if "in" in data['1']:
                    if data['alu_op'] in lut_supported_ops:
                        bit_inputs.add(data['1'])
                    else:
                        inputs.add(data['1'])

                if '2' in data:
                    if "in" in data['2'] and "mux" in data['alu_op']:
                        bit_inputs.add(data['2'])


            if "in" in data['0']:
                if data['alu_op'] in lut_supported_ops:
                    bit_inputs.add(data['0'])
                else:
                    inputs.add(data['0'])


    if last_eq == "":
        for node, data in ret_val.items():
            if data['alu_op'] != 'output' and data['alu_op'] != 'bit_output':
                last_eq = data['0']
                output_type = "Bit" if data['alu_op'] in bit_output_ops else "Data"

    print(last_eq)

    for i in inputs:
        args_str += i + " : Data, "
    for i in bit_inputs:
        args_str += i + " : Bit, "


    args_str = args_str[:-2]

    peak_output = '''
from peak import Peak, family_closure, Const
from peak import family
from peak.family import AbstractFamily

@family_closure
def mapping_function_fc(family: AbstractFamily):
    Data = family.BitVector[16]
    SData = family.Signed[16]
    Bit = family.Bit
    @family.assemble(locals(), globals())
    class mapping_function(Peak):
        def __call__(self, ''' + args_str + ''') -> ''' + output_type + ''':
            ''' + absd_str + '''
            return ''' + last_eq + '''
      
    return mapping_function
'''

    if not os.path.exists('outputs/peak_eqs'):
        os.makedirs('outputs/peak_eqs')

    with open("outputs/peak_eqs/peak_eq_" + str(sub_idx) + ".py", "w") as write_file:
        write_file.write(peak_output)

    return ret_val


def check_no_cycles(pair, g1, g1_map_r, g2, g2_map_r):

    a0 = g1_map_r[pair[0][1]['start']]
    a1 = g1_map_r[pair[1][1]['start']]

    b0 = g2_map_r[pair[0][1]['end']]
    b1 = g2_map_r[pair[1][1]['end']]


    apath = nx.algorithms.shortest_paths.generic.has_path(g1, a0, a1)
    apath_r = nx.algorithms.shortest_paths.generic.has_path(g1, a1, a0)
    bpath = nx.algorithms.shortest_paths.generic.has_path(g2, b0, b1)
    bpath_r = nx.algorithms.shortest_paths.generic.has_path(g2, b1, b0)

    cycle_exists = (apath and bpath_r) or (apath_r and bpath)

    # if cycle_exists:
    #     breakpoint()

    return not cycle_exists
    

def gen_verilog():
    arch = read_arch("outputs/subgraph_archs/subgraph_arch_merged.json")
    PE_fc = pe_arch_closure(arch)
    PE = PE_fc(family.MagmaFamily())

    if not os.path.exists('outputs/verilog'):
        os.makedirs('outputs/verilog')
    m.compile(f"outputs/verilog/PE", PE, output="coreir-verilog")