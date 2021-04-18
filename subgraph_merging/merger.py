import networkx as nx
import typing as tp
from itertools import count, combinations
import pulp

from .utils import *
import subgraph_merging.config as config
from .subgraph import Subgraph, DSESubgraph
from .plot_utils import *

class Merger():
    def __init__(self, subgraphs: tp.List):
        self.subgraphs = subgraphs

    def construct_compatibility_bipartite_graph(self, g0: nx.MultiDiGraph, g1: nx.MultiDiGraph):
        gb = nx.Graph()
       
        for n1 in g0.nodes():
            gb.add_node(
                n1,
                type='node',
                op=g0.nodes[n1]['op'],
                bipartite=0)

        for (u, v, p) in g0.edges.data('port'):
            gb.add_node(
                u + '/' + v,
                type='edge',
                node0=u,
                node1=v,
                op0=g0.nodes[u]['op'],
                op1=g0.nodes[v]['op'],
                port=p,
                bipartite=0)

        for n1 in g1.nodes():
            gb.add_node(
                n1,
                type='node',
                op=g1.nodes[n1]['op'],
                bipartite=1)

        for (u, v, p) in g1.edges.data('port'):
            gb.add_node(
                u + '/' + v,
                type='edge',
                node0=u,
                node1=v,
                op0=g1.nodes[u]['op'],
                op1=g1.nodes[v]['op'],
                port=p,
                bipartite=1)


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
                            if config.op_types[d1['op1']] in config.comm_ops:
                                gb.add_edge(n0, n1)

        return gb

    def construct_compatibility_graph(self, gb: nx.MultiDiGraph, g0: nx.MultiDiGraph, g1: nx.MultiDiGraph):
        weights = {}

        for k, v in config.op_types_flipped.items():
            if k in config.op_map and config.op_map[k] in config.op_costs:
                weights[str(v)] = config.op_costs[config.op_map[k]]["area"]
            else:
                weights[str(v)] = 1
                print(f"Node {k} not in op_costs")

        gc = nx.Graph()

        for i, (u, v) in enumerate(gb.edges):
            if gb.nodes.data(True)[u]['type'] == 'node':
                weight = weights[gb.nodes.data(True)[u]['op']]
                in_or_out = is_node_input_or_output(gb.nodes.data(True)[u])
                gc.add_node(i, start=u, end=v, weight=weight, in_or_out=in_or_out)
            else:
                weight = 30
                in_or_out = is_node_input_or_output(gb.nodes.data(True)[u])
                gc.add_node(i, start=u, end=v, weight=weight, in_or_out=in_or_out, start_port=gb.nodes.data(True)[u]["port"], end_port=gb.nodes.data(True)[v]["port"])

        for pair in combinations(gc.nodes.data(True), 2):
            
            if "/" in pair[0][1]['start'] and "/" in pair[1][1]['start']:
                start_0_0 = pair[0][1]['start'].split("/")[0]
                start_0_1 = pair[0][1]['start'].split("/")[1]
                start_1_0 = pair[1][1]['start'].split("/")[0]
                start_1_1 = pair[1][1]['start'].split("/")[1]

                end_0_0 = pair[0][1]['end'].split("/")[0]
                end_0_1 = pair[0][1]['end'].split("/")[1]
                end_1_0 = pair[1][1]['end'].split("/")[0]
                end_1_1 = pair[1][1]['end'].split("/")[1]

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

            elif "/" in pair[0][1]['start']:
                start_0_0 = pair[0][1]['start'].split("/")[0]
                start_0_1 = pair[0][1]['start'].split("/")[1]
                start_1 = pair[1][1]['start']

                end_0_0 = pair[0][1]['end'].split("/")[0]
                end_0_1 = pair[0][1]['end'].split("/")[1]
                end_1 = pair[1][1]['end']

                if start_0_0 == start_1:
                    if end_0_0 == end_1:
                        gc.add_edge(pair[0][0], pair[1][0])
                elif start_0_1 == start_1:
                    if end_0_1 == end_1:
                        gc.add_edge(pair[0][0], pair[1][0])
                elif not (end_0_0 == end_1 or end_0_1 == end_1):
                    gc.add_edge(pair[0][0], pair[1][0])

            elif "/" in pair[1][1]['start']:
                start_0 = pair[0][1]['start']
                start_1_0 = pair[1][1]['start'].split("/")[0]
                start_1_1 = pair[1][1]['start'].split("/")[1]

                end_0 = pair[0][1]['end']
                end_1_0 = pair[1][1]['end'].split("/")[0]
                end_1_1 = pair[1][1]['end'].split("/")[1]

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
                    if check_no_cycles(pair, g0, g1):
                        gc.add_edge(pair[0][0], pair[1][0])
            elif not (pair[0][1]['end'] == pair[1][1]['end']):
                if check_no_cycles(pair, g0, g1):
                    gc.add_edge(pair[0][0], pair[1][0])

        return gc


    def find_maximum_weight_clique(self, gc: nx.MultiDiGraph):
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

        return C

    def reconstruct_merged_graph(self, c: nx.MultiDiGraph, g0: nx.MultiDiGraph, g1: nx.MultiDiGraph):
        g = g0.copy()

        for k, v in c:
            if "/" in k:
                g0u = k.split("/")[0]
                g0v = k.split("/")[1]
                g1u = v.split("/")[0]
                g1v = v.split("/")[1]
                if g0.edges[(g0u, g0v, 0)]['port'] != g1.edges[(g1u, g1v, 0)]['port'] and (g0u, g0v, 1) not in g0.edges:
                    if config.op_types[g1.nodes.data(True)[g1v]['op_config'][0]] in config.comm_ops:
                        swap_ports(g1, g1v)
                    else:
                        print("Oops, something went wrong")
                        exit()

        b = {j:i for i,j in c}

        for n, d in g1.nodes.data(True):
            if n not in b:
                g.add_node(str(config.node_counter), op=d['op'], op_config=d['op_config'])
                b[n] = str(config.node_counter)
                config.node_counter += 1

        for u, v, d in g1.edges.data(True):
            if not str(u) + "/" + str(v) in b and not g.has_edge(b[u],b[v]):
                g.add_edge(b[u], b[v], port=d['port'], regs=0)

        relabel = {i:j for i,j in b.items() if "/" not in i}
        nx.relabel_nodes(g1, relabel, copy=False)
        return DSESubgraph(g)

    def set_merged_graph(self, merged_graph: nx.MultiDiGraph):
        self.merged_graph = merged_graph

    def merge_subgraphs(self, g0: nx.MultiDiGraph, g1: nx.MultiDiGraph):
        gb = self.construct_compatibility_bipartite_graph(g0, g1)
        gc = self.construct_compatibility_graph(gb, g0, g1)
        # plot_compatibility_graph(g0, g1, gb, gc)
        c = self.find_maximum_weight_clique(gc)
        return self.reconstruct_merged_graph(c, g0, g1)

    def merge_all_subgraphs(self):
        merged_graph = self.subgraphs[0]
        for subgraph in self.subgraphs[1:]:
            merged_graph = self.merge_subgraphs(merged_graph.subgraph, subgraph.subgraph)

        self.set_merged_graph(merged_graph)

    def print_area_and_energy(self):
        energy_area_dict = self.merged_graph.calc_area_and_energy()
        print("Total energy:", energy_area_dict["energy"])
        print("Total area:", energy_area_dict["area"])


class DSEMerger(Merger):
    def __init__(self, subgraphs: tp.List):
        super().__init__(subgraphs)

    def merged_graph_to_arch(self):
        self.merged_graph.generate_peak_arch()

    def write_merged_graph_arch(self):
        self.merged_graph.write_peak_arch("outputs/PE.json")

    def generate_rewrite_rules(self):
        for idx, subgraph in enumerate(self.subgraphs):
            subgraph.generate_rewrite_rule(idx)

    def write_rewrite_rules(self):
        for idx, subgraph in enumerate(self.subgraphs):
            subgraph.write_rewrite_rule("outputs/rewrite_rules/rewrite_rule_" + str(idx) + ".json")