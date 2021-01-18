import os
import json
import importlib
import typing as tp
import networkx as nx
import matplotlib.pyplot as plt
from itertools import count, combinations

from pysmt.logics import QF_BV
import subgraph_merging.config as config
import subgraph_merging.utils as utils
from peak.mapper import ArchMapper
from peak_gen.peak_wrapper import wrapped_peak_class
from peak_gen.arch import read_arch, graph_arch


class Subgraph():
    def __init__(self, subgraph: nx.MultiDiGraph):
        self.subgraph = subgraph


class DSESubgraph(Subgraph):
    def __init__(self, subgraph: nx.MultiDiGraph):
        super().__init__(subgraph)

    def add_input_and_output_nodes(self):

        for n, d in self.subgraph.copy().nodes.data(True):
            bitwidth = config.op_bitwidth[config.op_types[d["op_config"][0]]]

            pred = set()
            for s in self.subgraph.pred[n]:
                pred.add(self.subgraph.edges[(s, n, 0)]['port'])

            for i, bitw in enumerate(bitwidth):
                if str(i) not in pred:
                    if bitw == 1:
                        self.subgraph.add_node(str(config.node_counter), op="bit_input", op_config=["bit_input"])
                        self.subgraph.add_edge(str(config.node_counter), n, port=str(i))
                    else:
                        self.subgraph.add_node(str(config.node_counter), op="input", op_config=["input"])
                        self.subgraph.add_edge(str(config.node_counter), n, port=str(i))               

                    config.node_counter += 1

            if len(bitwidth) == 0: #const input
                if bitwidth == 1:
                    self.subgraph.add_node(
                        str(config.node_counter),
                        op="bit_const_input",
                        op_config=["bit_const_input"])
                    self.subgraph.add_edge(str(config.node_counter), n, port='0')
                else:
                    self.subgraph.add_node(
                        str(config.node_counter),
                        op="const_input",
                        op_config=["const_input"])
                    self.subgraph.add_edge(str(config.node_counter), n, port='0')
                config.node_counter += 1

        for n, d in self.subgraph.copy().nodes.data(True):
            if len(list(self.subgraph.successors(n))) == 0:
                if config.op_types[d["op_config"][0]] in config.bit_output_ops:
                    self.subgraph.add_node(str(config.node_counter), op="bit_output", op_config=["bit_output"])
                    self.subgraph.add_edge(n, str(config.node_counter), port='0')
                else:
                    self.subgraph.add_node(str(config.node_counter), op="output", op_config=["output"])
                    self.subgraph.add_edge(n, str(config.node_counter), port='0')
                config.node_counter += 1

    def generate_peak_eq(self):
        absd_count = 0
        pred_dict = {}
        input_nodes = set()

        for n, d in self.subgraph.nodes.data(True):
            if utils.is_node_input(d):
                input_nodes.add("data" + n)
            else:
                pred = {}
                pred['op_config'] = config.op_types[self.subgraph.nodes[n]['op_config'][0]]
                for s in self.subgraph.pred[n]:
                    pred["port"+self.subgraph.edges[(s, n, 0)]['port']] = "data" + s
                pred_dict["data" + n] = pred

        # print(pred_dict)

        pred_dict_copy = pred_dict.copy()
        args_str = ""
        eq_dict = {}
        inputs = set()
        bit_inputs = set()
        absd_str = ""
        last_eq = ""
        output_type = ""

        while len(pred_dict) > 0:
            for node, data in pred_dict.copy().items():
                if data['op_config'] == 'output' or data['op_config'] == 'bit_output':
                    pred_dict.pop(node)
                elif data['op_config'] == 'const':
                    eq_dict[node] = data['port0']
                    pred_dict.pop(node)
                    args_str += data['port0'] + " : Const(Data), "
                elif data['op_config'] == 'bitconst':
                    eq_dict[node] = data['port0']
                    pred_dict.pop(node)
                    args_str += data['port0'] + " : Const(Bit), "
                
                else:
                    if (data['port0'] in input_nodes or data['port0'] in eq_dict) and (data['port1'] in input_nodes or data['port1'] in eq_dict):

                        if 'port2' in data:
                            if (data['port2'] in input_nodes or data['port2'] in eq_dict):
                                eq_dict[node], absd_str_tmp, absd_count = utils.construct_eq(
                                str(eq_dict.get(data['port0'], data['port0'])),
                                str(eq_dict.get(data['port1'], data['port1'])), data['op_config'], absd_count,
                                str(eq_dict.get(data['port2'], data['port2'])))
                                last_eq = eq_dict[node]
                                output_type = "Bit" if data['op_config'] in config.bit_output_ops else "Data"
                                pred_dict.pop(node)
                                absd_str += absd_str_tmp
                        else:
                            eq_dict[node], absd_str_tmp, absd_count = utils.construct_eq(
                                str(eq_dict.get(data['port0'], data['port0'])),
                                str(eq_dict.get(data['port1'], data['port1'])), data['op_config'], absd_count)

                            absd_str += absd_str_tmp
                            last_eq = eq_dict[node]
                            output_type = "Bit" if data['op_config'] in config.bit_output_ops else "Data"
                            pred_dict.pop(node)
                    if data['port0'] in input_nodes:
                        if data['op_config'] in config.lut_supported_ops:
                            bit_inputs.add(data['port0'])
                        else:
                            inputs.add(data['port0'])
                    if data['port1'] in input_nodes:
                        if data['op_config'] in config.lut_supported_ops:
                            bit_inputs.add(data['port1'])
                        else:
                            inputs.add(data['port1'])

                    if 'port2' in data:
                        if data['port2'] in input_nodes and "mux" in data['op_config']:
                            bit_inputs.add(data['port2'])


        if last_eq == "":
            for node, data in pred_dict_copy.items():
                if data['op_config'] != 'output' and data['op_config'] != 'bit_output':
                    last_eq = data['port0']
                    output_type = "Bit" if data['op_config'] in config.bit_output_ops else "Data"

        # print(last_eq)

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

        self.peak_eq = peak_output
        self.short_eq = last_eq


    def write_peak_eq(self, filename: str):
        if not hasattr(self, "peak_eq"):
            raise ValueError("Generate peak eq first")

        if not os.path.exists('outputs/peak_eqs'):
            os.makedirs('outputs/peak_eqs')

        with open(filename, "w") as write_file:
            write_file.write(self.peak_eq)

    def generate_rewrite_rule(self, subgraph_ind):

        if not os.path.exists("./outputs/PE.json"):
            raise ValueError("Generate and write merged graph peak arch first")

        if not os.path.exists("./outputs/peak_eqs/peak_eq_" + str(subgraph_ind) + ".py"):
            raise ValueError("Generate and write peak_eq first")

        arch = read_arch("./outputs/PE.json")
        graph_arch(arch)
        PE_fc = wrapped_peak_class(arch, debug=False)
        arch_mapper = ArchMapper(PE_fc)

        peak_eq = importlib.import_module("outputs.peak_eqs.peak_eq_" + str(subgraph_ind))

        print("\nGenerating rewrite rule for:")
        print(self.short_eq)
        ir_mapper = arch_mapper.process_ir_instruction(peak_eq.mapping_function_fc)
        solution = ir_mapper.solve(external_loop = True)
        if solution is None:
            utils.print_red("No rewrite rule found")
        else:
            utils.print_green("Found rewrite rule")
            self.rewrite_rule = solution

    def write_rewrite_rule(self, filename: str):
        if not hasattr(self, "rewrite_rule"):
            raise ValueError("Generate rewrite rule first")

        serialized_rr = self.rewrite_rule.serialize_bindings()

        if not os.path.exists('outputs/rewrite_rules'):
            os.makedirs('outputs/rewrite_rules')

        with open(filename, "w") as write_file:
            json.dump(serialized_rr, write_file, indent=2)

    def generate_peak_arch(self):
        arch = {}
        arch["input_width"] = 16
        arch["output_width"] = 16
        arch["enable_input_regs"] = False
        arch["enable_output_regs"] = False
        modules = {}
        ids = []
        connected_ids = []

        for n, d in self.subgraph.nodes.data(True):

            # Don't want to put input and output nodes in actual arch
            if d["op"] != "output" and d["op"] != "bit_output" and d["op"] != "const_input" and d["op"] != "bit_const_input":

                # Only want id of input nodes
                if d["op"] != "input" and d["op"] != "bit_input":
                    modules[n] = {}
                    modules[n]["id"] = n

                    op = config.op_types[d["op"]]
                    op_config = [config.op_types[x] for x in d["op_config"]]

                    modules[n]["type"] = op

                ids.append(n)

        outputs = set()
        bit_outputs = set()

        for u, v, d in self.subgraph.edges.data(True):
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
            if self.subgraph.nodes.data(True)[v]["op"] == "output":
                outputs.add(u)

            if self.subgraph.nodes.data(True)[v]["op"] == "bit_output":
                bit_outputs.add(u)

        arch["modules"] = [v for v in modules.values()]
        arch["outputs"] = [list(outputs)]
        if len(bit_outputs) > 0:
            arch["bit_outputs"] = [list(bit_outputs)]
        else:
            arch["bit_outputs"] = []

        arch["modules"] = utils.sort_modules(arch["modules"], self.subgraph)
        self.arch = arch


    def write_peak_arch(self, filename: str):
        if not hasattr(self, "arch"):
            raise ValueError("Generate peak arch first")

        if not os.path.exists('outputs/'):
            os.makedirs('outputs/')

        with open(filename, "w") as write_file:
            write_file.write(json.dumps(self.arch, indent=4, sort_keys=True))

    def plot(self):
        ret_g = self.subgraph.copy()

        groups = set(nx.get_node_attributes(ret_g, 'op').values())
        mapping = dict(zip(sorted(groups), count()))
        nodes = ret_g.nodes()
        colors = [mapping[ret_g.nodes[n]['op']] for n in nodes]
        labels = {}
        edge_labels = {}
        for n in nodes:
            labels[n] = config.op_types[ret_g.nodes[n]['op']] + "\n" + n
        for u, v, d in ret_g.edges(data = True):
            edge_labels[(u,v)] = d["port"]
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
            # node_list=nodes,
            node_color=colors,
            # with_labels=False,
            node_size=1500,
            cmap=plt.cm.Pastel1,
            alpha=1)
        nx.draw_networkx_labels(ret_g, pos, labels)
        nx.draw_networkx_edge_labels(ret_g,pos,edge_labels=edge_labels)

        plt.show()