import os
import json
import copy
import shutil 
import ast
import networkx as nx
import pickle


primitive_ops = [
    "and", "or", "xor", "shl", "lshr", "ashr", "add", "sub",
    "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", "smax", "smin", "umax", "umin", "absd", "abs", "mul", "mux"
]

alu_supported_ops = [
    "and", "or", "xor", "shl", "lshr", "ashr", "add", "sub",
    "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", "smax", "smin", "umax", "umin", "absd", "abs"
]


def read_subgraphs(file_ind_pairs, op_types):

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
                if op_types[line.split()[2]] in alu_supported_ops:
                    op = "-1"
                else:
                    op = line.split()[2]
                graphs_per_file[graph_ind].add_node(
                    line.split()[1], op=op, alu_ops=[line.split()[2]])
                graphs_sizes[graph_ind] += 1
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


    op_types["-1"] = "alu"
    op_types["input"] = "input"
    op_types["bit_input"] = "bit_input"
    op_types["const_input"] = "const_input"
    op_types["output"] = "output"

    op_types_flipped = {v: k for k, v in op_types.items()}

    return op_types, op_types_flipped

def add_primitive_ops(graphs, op_types_flipped):

    with open(".temp/used_ops.txt", "rb") as file:
        used_ops = pickle.load(file)

    print(used_ops)

    for op in used_ops:
        if op != "const":
            op_graph = nx.MultiDiGraph()
            if op in alu_supported_ops:
                op_graph.add_node('0', op='-1', alu_ops=[op_types_flipped[op]])
            else:
                op_graph.add_node('0', op=op_types_flipped[op], alu_ops=[op_types_flipped[op]])
            graphs.append(op_graph)




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
    while len(modules) > 0:
        for module in modules.copy():
            if module['type'] == 'const':
                ids.append(module["id"])
                output_modules.append(module)
                modules.remove(module)
            else:
            
                inorder = True

                for in0_item in module["in0"]:
                    inorder = inorder and ("in" in in0_item or in0_item in ids)

                for in1_item in module["in1"]:
                    inorder = inorder and ("in" in in1_item or in1_item in ids)

                if module['type'] == 'mux':
                    for sel_item in module["sel"]:
                        inorder = inorder and ("in" in sel_item or sel_item in ids)

                if inorder:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)

    return output_modules


def merged_subgraph_to_arch(subgraph, op_types):

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

    for n, d in subgraph.nodes.data(True):

        # Don't want to put input and output nodes in actual arch
        if d["op"] != "output" and d["op"] != "const_input":

            # Only want id of input nodes
            if d["op"] != "input" and d["op"] != "bit_input":
                modules[n] = {}
                modules[n]["id"] = n

                op = op_types[d["op"]]
                alu_ops = [op_types[x] for x in d["alu_ops"]]

                if op == "mul" or op == "const" or op == "mux":
                    modules[n]["type"] = op
                else:
                    modules[n]["type"] = 'alu'

                    for alu_op in alu_ops:
                        if alu_op not in alu_supported_ops:
                            print("Warning: possible unsupported ALU operation found in subgraph:", n)



            ids.append(n)

    outputs = set()
    bit_outputs = set()

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
                elif d["port"] == "2":
                    if "sel" in modules[v]:
                        modules[v]["sel"].append(u)
                    else:
                        modules[v]["sel"] = [u]
        
        # Add to output mux
        if subgraph.nodes.data(True)[v]["op"] == "output":
            outputs.add(u)

            alu_ops = [op_types[x] for x in subgraph.nodes.data(True)[u]["alu_ops"]]
            for alu_op in alu_ops:
                if alu_op in bit_output_ops:
                    bit_outputs.add(u)

    arch["modules"] = [v for v in modules.values()]
    arch["outputs"] = [list(outputs)]
    arch["bit_outputs"] = list(bit_outputs)

    if not os.path.exists('outputs/subgraph_archs'):
        os.makedirs('outputs/subgraph_archs')

    arch["modules"] = sort_modules(arch["modules"])

    with open("outputs/subgraph_archs/subgraph_arch_merged.json", "w") as write_file:
        write_file.write(json.dumps(arch, indent=4, sort_keys=True))

    return arch


def construct_eq(in0, in1, op, in2=""):

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
    op_str_map["sle"] = "(Bit(1) if SData(in_0) <= SData(in_1) else Bit(0))"
    op_str_map["sge"] = "(Bit(1) if SData(in_0) >= SData(in_1) else Bit(0))"
    op_str_map["ule"] = "(Bit(1) if in_0 <= in_1 else Bit(0))"
    op_str_map["uge"] = "(Bit(1) if in_0 >= in_1 else Bit(0))"
    op_str_map["eq"] = "(Bit(1) if in_0 == in_1 else Bit(0))"
    op_str_map["slt"] = "(Bit(1) if SData(in_0) < SData(in_1) else Bit(0))"
    op_str_map["sgt"] = "(Bit(1) if SData(in_0) > SData(in_1) else Bit(0))"
    op_str_map["ult"] = "(Bit(1) if in_0 < in_1 else Bit(0))"
    op_str_map["ugt"] = "(Bit(1) if in_0 > in_1 else Bit(0))"
    op_str_map["mux"] = "(in_0 if in_2 == Bit(0) else in_1)"
    op_str_map["umax"] = "(in_0 if in_0 > in_1 else in_1)"
    op_str_map["umin"] = "(in_0 if in_0 < in_1 else in_1)"
    op_str_map["smax"] = "(SData(in_0) if SData(in_0) > SData(in_1) else SData(in_1))"
    op_str_map["smin"] = "(SData(in_0) if SData(in_0) < SData(in_1) else SData(in_1))"
    op_str_map["abs"] = "(-in_0 if in_0 < Data(0) else in_0)"
    op_str_map["absd"] = "(in_0 - in_1)"

    return op_str_map[op].replace("in_0", in0).replace("in_1", in1).replace(
        "in_2", in2)


def subgraph_to_peak(subgraph, sub_idx, b, op_types):
    node_dict = {}

    bit_output_ops = {"sle", "sge", "slt", "sgt", "ule", "uge", "ult", "ugt", "eq"}

    for n, d in subgraph.nodes.data(True):
        if op_types[subgraph.nodes[n]['alu_ops'][0]] != "input" \
            and op_types[subgraph.nodes[n]['alu_ops'][0]] != "bit_input" \
            and op_types[subgraph.nodes[n]['alu_ops'][0]] != "const_input":

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
    last_eq = ""
    output_type = ""

    while len(node_dict) > 0:
        for node, data in node_dict.copy().items():
            if data['alu_op'] == 'output':
                node_dict.pop(node)
            elif data['alu_op'] == 'const':
                eq_dict[node] = data['0']
                node_dict.pop(node)
                args_str += data['0'] + " : Const(Data), "
            else:
                if ("in" in data['0'] or data['0'] in eq_dict) and ("in" in data['1'] or data['1'] in eq_dict):

                    if '2' in data:
                        if ("in" in data['2'] or data['2'] in eq_dict):
                            eq_dict[node] = construct_eq(
                            str(eq_dict.get(data['0'], data['0'])),
                            str(eq_dict.get(data['1'], data['1'])), data['alu_op'],
                            str(eq_dict.get(data['2'], data['2'])))
                            last_eq = eq_dict[node]
                            output_type = "Bit" if data['alu_op'] in bit_output_ops else "Data"
                            node_dict.pop(node)
                    else:
                        eq_dict[node] = construct_eq(
                            str(eq_dict.get(data['0'], data['0'])),
                            str(eq_dict.get(data['1'], data['1'])), data['alu_op'])


                        last_eq = eq_dict[node]
                        output_type = "Bit" if data['alu_op'] in bit_output_ops else "Data"
                        node_dict.pop(node)
                if "in" in data['1']:
                    inputs.add(data['1'])

                if '2' in data:
                    if "in" in data['2']:
                        bit_inputs.add(data['2'])


            if "in" in data['0']:
                inputs.add(data['0'])

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
  
            return ''' + last_eq + '''
      
    return mapping_function
'''

    if not os.path.exists('outputs/peak_eqs'):
        os.makedirs('outputs/peak_eqs')

    with open("outputs/peak_eqs/peak_eq_" + str(sub_idx) + ".py", "w") as write_file:
        write_file.write(peak_output)

    return ret_val