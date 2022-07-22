import os
import json
import time
import importlib
import math
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

import hwtypes


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
                        self.subgraph.add_edge(str(config.node_counter), n, port=str(i), regs=0)
                    else:
                        self.subgraph.add_node(str(config.node_counter), op="input", op_config=["input"])
                        self.subgraph.add_edge(str(config.node_counter), n, port=str(i), regs=0)               

                    config.node_counter += 1

            if len(bitwidth) == 0: #const input
                if config.op_types[d["op_config"][0]] in config.bit_output_ops:
                    self.subgraph.add_node(
                        str(config.node_counter),
                        op="bit_const_input",
                        op_config=["bit_const_input"])
                    self.subgraph.add_edge(str(config.node_counter), n, port='0', regs=0)
                else:
                    self.subgraph.add_node(
                        str(config.node_counter),
                        op="const_input",
                        op_config=["const_input"])
                    self.subgraph.add_edge(str(config.node_counter), n, port='0', regs=0)
                config.node_counter += 1

        for n, d in self.subgraph.copy().nodes.data(True):
            if len(list(self.subgraph.successors(n))) == 0:
                if config.op_types[d["op_config"][0]] in config.bit_output_ops:
                    self.subgraph.add_node(str(config.node_counter), op="bit_output", op_config=["bit_output"])
                    self.subgraph.add_edge(n, str(config.node_counter), port='0', regs=0)
                else:
                    self.subgraph.add_node(str(config.node_counter), op="output", op_config=["output"])
                    self.subgraph.add_edge(n, str(config.node_counter), port='0', regs=0)
                config.node_counter += 1

    def generate_peak_eq(self):
        absd_count = 0
        pred_dict = {}
        input_nodes = set()

        for n, d in self.subgraph.nodes.data(True):
            if utils.is_node_input_or_const(d):
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
    Data32 = family.Unsigned[32]
    SInt = family.Signed[16]
    UInt = family.Unsigned[16]
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

    def contains_mul(self):
        for n, d in self.subgraph.nodes.data(True):
            op = config.op_types[d['op']]
            if op == "mul" or op == "mult_middle":
                return True

        return False

    def write_peak_eq(self, filename: str):
        if not hasattr(self, "peak_eq"):
            raise ValueError("Generate peak eq first")

        if not os.path.exists('outputs/peak_eqs'):
            os.makedirs('outputs/peak_eqs')

        with open(filename, "w") as write_file:
            write_file.write(self.peak_eq)

    def generate_rewrite_rule(self, subgraph_ind, mul_op):

        if not os.path.exists("./outputs/PE.json"):
            raise ValueError("Generate and write merged graph peak arch first")

        if not os.path.exists("./outputs/peak_eqs/peak_eq_" + str(subgraph_ind) + ".py"):
            raise ValueError("Generate and write peak_eq first")
        print("\nGenerating rewrite rule for:")
        print(self.short_eq)

        arch = read_arch("./outputs/PE.json")
        graph_arch(arch)
        PE_fc = wrapped_peak_class(arch)

        arch_inputs = arch.inputs

        input_constraints = {}

        for n, d in self.subgraph.nodes.data(True):
            if utils.is_node_input(d):
                if utils.is_node_bit_input(d):
                    input_constraints[(f"bitinputs{arch.bit_inputs.index(n)}",)] = (f"data{n}",)
                else:
                    input_constraints[(f"inputs{arch.inputs.index(n)}",)] = (f"data{n}",)

        path_constraints = {}

        if not mul_op:
            print("Datagating multipliers")
            idx = 0
            for module in arch.modules:
                if module.type_ == "mul":
                    path_constraints[('inst', 'mul', idx)] = hwtypes.smt_bit_vector.SMTBitVector[2](1)
                    print(path_constraints)
                    idx += 1

        # arch_mapper = ArchMapper(PE_fc, path_constraints = path_constraints, input_constraints = input_constraints)
        arch_mapper = ArchMapper(PE_fc, path_constraints = path_constraints)
        peak_eq = importlib.import_module("outputs.peak_eqs.peak_eq_" + str(subgraph_ind))

        if subgraph_ind > -1:
            print("Solving...")
            ir_mapper = arch_mapper.process_ir_instruction(peak_eq.mapping_function_fc, simple_formula=False)
            start = time.time()
            solution = ir_mapper.solve('btor', external_loop=True, itr_limit=80, logic=QF_BV)
            # solution = ir_mapper.solve('z3')
            end = time.time()
            print("Rewrite rule solving time:", end-start)
        else:
            print("Skipping...")
            solution = None

        if solution is None:
            utils.print_red("No rewrite rule found, trying without input constraints")
            arch_mapper = ArchMapper(PE_fc)
            ir_mapper = arch_mapper.process_ir_instruction(peak_eq.mapping_function_fc)
            solution = ir_mapper.solve()
            if solution is None:
                print("Still couldn't find solution")
                exit()
            else:
                utils.print_green("Found rewrite rule")
                self.rewrite_rule = solution
        else:
            utils.print_green("Found rewrite rule")
            self.rewrite_rule = solution
            #counter_ex = solution.verify()
            #assert counter_ex == None
            #for i in solution.ibinding: print(i)

    def write_rewrite_rule(self, filename: str):
        if not hasattr(self, "rewrite_rule"):
            raise ValueError("Generate rewrite rule first")

        serialized_rr = self.rewrite_rule.serialize_bindings()

        serialized_rr["ibinding"][-1] = ({'type': 'Bit', 'width': 1, 'value': True},  ('clk_en',))

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
        regs_counter = 0
        for u, v, d in self.subgraph.edges.data(True):
            u_temp = u
            v_temp = v
            for r in range(d["regs"]):
                name = "reg" + str(regs_counter)
                modules[name] = {}
                modules[name]["id"] = name
         
                if config.op_types[self.subgraph.nodes.data(True)[v]["op"]] == "lut" or \
                    self.subgraph.nodes.data(True)[v]["op"] == "bit_output" or \
                    self.subgraph.nodes.data(True)[v]["op"] == "bit_const" or \
                    d['port'] == '2':
                    modules[name]["type"] = "bitreg"
                else:
                    modules[name]["type"] = "reg"
                modules[name]["in"] = [u_temp]
                u_temp = name
                ids.append(name)
                regs_counter += 1
            if v_temp in modules:
                connected_ids.append(u_temp)
                if modules[v_temp]["type"] != "const" and modules[v_temp]["type"] != "bitconst":
                    if d["port"] == "0":
                        if "in0" in modules[v_temp]:
                            modules[v_temp]["in0"].append(u_temp)
                        else:
                            modules[v_temp]["in0"] = [u_temp]
                    elif d["port"] == "1":
                        if "in1" in modules[v_temp]:
                            modules[v_temp]["in1"].append(u_temp)
                        else:
                            modules[v_temp]["in1"] = [u_temp]
                    elif d["port"] == "2":
                        if "in2" in modules[v_temp]:
                            modules[v_temp]["in2"].append(u_temp)
                        else:
                            modules[v_temp]["in2"] = [u_temp]
            
            # Add to output mux
            if self.subgraph.nodes.data(True)[v_temp]["op"] == "output":
                outputs.add(u_temp)

            if self.subgraph.nodes.data(True)[v_temp]["op"] == "bit_output":
                bit_outputs.add(u_temp)

        arch["modules"] = [v for v in modules.values()]
        arch["outputs"] = [list(outputs)]
        if len(bit_outputs) > 0:
            arch["bit_outputs"] = [list(bit_outputs)]
        else:
            arch["bit_outputs"] = []
        arch["modules"] = utils.sort_modules(arch["modules"], self.subgraph)
        self.arch = arch


    def write_peak_arch(self, filename: str, input_regs=False):
        if not hasattr(self, "arch"):
            raise ValueError("Generate peak arch first")

        if not os.path.exists('outputs/'):
            os.makedirs('outputs/')

        if input_regs:
            self.arch["enable_input_regs"] = True

        with open(filename, "w") as write_file:
            write_file.write(json.dumps(self.arch, indent=4, sort_keys=True))

        arch = read_arch("./outputs/PE.json")
        graph_arch(arch)

    def analyze_pe(self):
        if not hasattr(self, "arch"):
            raise ValueError("Generate peak arch first")

        arch = read_arch("./outputs/PE.json")
        arch_stats = {}
        arch_stats['num_inputs'] = arch.num_inputs
        arch_stats['num_bit_inputs'] = arch.num_bit_inputs
        arch_stats['num_outputs'] = arch.num_outputs
        arch_stats['num_bit_outputs'] = arch.num_bit_outputs
        arch_stats['num_modules'] = len(arch.modules)
        arch_stats['num_reg'] = arch.num_reg 
        arch_stats['num_bit_reg'] = arch.num_bit_reg 

        arch_stats['num_IO'] = arch.num_inputs + arch.num_outputs 

        inputs = set()
        outputs = set()
        for n,d in self.subgraph.nodes.data(True): 
            if "input" in d['op']:
                inputs.add(n) 

            if "output" in d['op']:
                outputs.add(n) 

        total_paths = 0
        for iput in inputs:
            for oput in outputs:
                total_paths += len(list(nx.all_simple_paths(self.subgraph, source=iput, target=oput)))

        print("PE stats")
        print("Num ops:", arch_stats['num_modules'])
        print("Num I/O:", arch_stats['num_IO'])
        print("Num paths:", total_paths)

    def plot(self, index):
        plt.clf()
        ret_g = self.subgraph.copy()
        groups = set(nx.get_node_attributes(ret_g, 'op').values())
        mapping = dict(zip(sorted(groups), count()))
        nodes = ret_g.nodes()
        colors = [mapping[ret_g.nodes[n]['op']] for n in nodes]
        labels = {}
        edge_labels = {}
        for n in nodes:
            if ret_g.nodes[n]['op'] in config.op_types:
                labels[n] = config.op_types[ret_g.nodes[n]['op']] + "\n" + n
            else:
                labels[n] = ret_g.nodes[n]['op']
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

        plt.savefig(f"{index}.png")
    
    def pipeline(self, num_regs):
        # self.plot()
        g = self.subgraph
        g.add_node("start", op="start")

        for n, d in g.copy().nodes(data = True):
            if utils.is_node_input_or_const(d):
                g.add_edge("start", n, regs=0)

        g.add_node("end", op="end")

        arrival_time = {}
        for n, d in g.copy().nodes(data = True):
            arrival_time[n] = 0
            if utils.is_node_output(d):
                g.add_edge(n, "end", regs=0)

        queue = []
        queue.append("start")

        while len(queue) > 0:
            curr_node = queue.pop(0)
            for succ in g.successors(curr_node):
                queue.append(succ)
                succ_node = g.nodes[succ]
                if succ_node['op'] in config.op_types and config.op_types[succ_node['op']] in config.op_map:
                    if arrival_time[succ] < arrival_time[curr_node] + config.op_costs[config.op_map[config.op_types[succ_node['op']]]]["crit_path"]:
                        arrival_time[succ] = arrival_time[curr_node] + config.op_costs[config.op_map[config.op_types[succ_node['op']]]]["crit_path"]
                else:
                    if arrival_time[succ] < arrival_time[curr_node]:
                        arrival_time[succ] = arrival_time[curr_node]

        print("Arrival time:")
        print(arrival_time, "\n")


        critical_path = []

        curr_node = "end"
        while curr_node != "start":
            max_time = -1 
            for pred in g.predecessors(curr_node):
                if arrival_time[pred] > max_time:
                    max_time = arrival_time[pred]
                    max_pred = pred
            curr_node = max_pred
            critical_path.append((curr_node, max_time))

        print("Critical Path:", critical_path)


        regs = {}
        for edge in g.edges:
            if utils.is_node_output(g.nodes[edge[1]]):
                regs[edge] = {"regs":num_regs}
            else:
                regs[edge] = {"regs":0}

        nx.set_edge_attributes(g, regs)

        t_min = 0
        for n in g.nodes():
            t_min = max(t_min, utils.get_node_timing(g, n))

        t_max = utils.get_clock_period(g)

        i = 0
        while (t_max - t_min) > 0.01:
            print("Iteration", i)
            print("t_min:", t_min)
            print("t_max:", t_max)
            t = (t_min + t_max)/2
            if self.retime(t):
                t_max = utils.get_clock_period(g)
            else:
                t_min += 0.01
                t_max = max(t_max, utils.get_clock_period(g))
            i += 1
            print()

        # self.retime(1.2)
        print("Final clock period:", utils.get_clock_period(g))

        g.remove_node("start")
        g.remove_node("end")

        # self.plot()
        # breakpoint()
        # exit()
      

    def retime(self, T):
        print("Retiming with T:", T)
        g = self.subgraph

        nodes = list(reversed(list(nx.topological_sort(g))))
        for v in nodes:
            if v != 'start' and v != 'end':
                successors = utils.successors(g, v)
                if len(successors) == 0:
                    n = 0
                else:
                    n = g[v][successors[0]].get(0)['regs']
                for succ in successors:
                    new_n = g[v][succ].get(0)['regs']
                    if new_n < n:
                        n = new_n
                for e in utils.successors(g, v):
                    g[v][e].get(0)['regs'] -= n
                for e in utils.predecessors(g, v):
                    g[e][v].get(0)['regs'] += n
        
        # self.plot()
        # return True
        delta = {}
        delta['start'] = 0
        for v in list(nx.topological_sort(g)):
            if v != 'start' and v != 'end':
                predecessors = utils.predecessors(g, v)
                if len(predecessors) == 0:
                    n = 0
                else:
                    n = g[predecessors[0]][v].get(0)['regs']
                for pred in predecessors:
                    new_n = g[pred][v].get(0)['regs']
                    if new_n < n:
                        n = new_n

                max_delta = 0
                for pred in predecessors:
                    if g[pred][v].get(0)['regs'] == n:
                        new_delta = delta[pred] 
                        if new_delta > max_delta:
                            max_delta = new_delta

                v_cost = utils.get_node_timing(g, v)
                delta[v] = v_cost + max_delta

                if delta[v] > T:
                    if n == 0:
                        return False
                    
                    n -= 1
                    delta[v] = utils.get_node_timing(g, v)
                
                if 'end' not in utils.successors(g, v):
                    for e in utils.predecessors(g, v):
                        g[e][v].get(0)['regs'] -= n
                    for e in utils.successors(g, v):
                        g[v][e].get(0)['regs'] += n

        return True

    def calc_area_and_energy(self):
        energy_area_dict = {}
        energy_area_dict["energy"] = 0
        energy_area_dict["area"] = 0
        for node, data in self.subgraph.nodes(data=True):
            op = config.op_types[data["op"]]
            if op in config.op_costs:
                energy_area_dict["energy"] += config.op_costs[op]["energy"]
                energy_area_dict["area"] += config.op_costs[op]["area"]

        return energy_area_dict
