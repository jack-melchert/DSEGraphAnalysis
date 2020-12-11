import networkx as nx
import typing as tp

class Subgraph():
    def __init__(self, subgraph: nx.MultiDiGraph):
        self.subgraph = subgraph

    def set_node_mapping(self, mapping_to: str, mapping: tp.Dict):
        self.mapping_to = mapping_to
        self.mapping = mapping

    def get_node_mapping(self):
        return self.mapping_to, self.mapping


class DSESubgraph(Subgraph):
    def __init__(self, subgraph: nx.MultiDiGraph):
        super().__init__(subgraph)

    def get_peak_eq(self):
        pass

    def write_peak_eq(self, peak_eq: str, filename: str):
        pass

    def get_rewrite_rule(self):
        pass

    def write_rewrite_rule(self, rewrite_rule: str, filename: str):
        pass

    def get_peak_arch(self):
        pass

    def write_peak_arch(self, arch: tp.Dict, filename: str):
        pass