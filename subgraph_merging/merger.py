import networkx as nx
import typing as tp


class Merger():
    def __init__(self, subgraphs: tp.List):
        self.subgraphs = subgraphs

    def construct_compatibility_graph(self, g0: nx.MultiDiGraph, g1: nx.MultiDiGraph):
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
                op1=g0.nodes[u]['op'],
                port=p,
                bipartite=0)

        for n1 in g1.nodes():
            gb.add_node(
                n1,
                type='node',
                op=g1.nodes[n1]['op'],
                bipartite=0)

        for (u, v, p) in g1.edges.data('port'):
            gb.add_node(
                u + '/' + v,
                type='edge',
                node0=u,
                node1=v,
                op0=g1.nodes[u]['op'],
                op1=g1.nodes[u]['op'],
                port=p,
                bipartite=0)


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
                            if config.op_types[d1['op1'][0]] in config.comm_ops:
                                gb.add_edge(n0, n1)

        return gb

    def find_maximum_weight_clique(self, gc: nx.MultiDiGraph):
        return None

    def reconstruct_merged_graph(self, c: nx.MultiDiGraph):
        return None

    def set_merged_graph(self, merged_graph: nx.MultiDiGraph):
        self.merged_graph = merged_graph

    def merge_subgraphs(self, g0: nx.MultiDiGraph, g1: nx.MultiDiGraph):
        gc = self.construct_compatibility_graph(g0, g1)
        c = self.find_maximum_weight_clique(gc)
        return self.reconstruct_merged_graph(c)

    def merge_all_subgraphs(self):
        merged_graph = self.subgraphs[0]
        for subgraph in self.subgraphs[1:]:
            merged_graph = self.merge_subgraphs(merged_graph.subgraph, subgraph.subgraph)

        self.set_merged_graph(merged_graph)


class DSEMerger(Merger):
    def __init__(self, subgraphs: tp.List):
        super().__init__(subgraphs)

    def merged_graph_to_arch(self):
        pass

    def write_merged_graph_arch(self):
        pass