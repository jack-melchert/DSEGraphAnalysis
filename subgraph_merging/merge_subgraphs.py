import pickle

import subgraph_merging.utils as utils
from .merger import DSEMerger 
import subgraph_merging.config as config

def merge_subgraphs(file_ind_pairs):
    utils.clean_output_dirs()

    with open(".temp/op_types.txt", "rb") as file:
        op_types_from_file = pickle.load(file)

    curr_ops = [*op_types_from_file]

    for op in config.primitive_ops:
        if op not in curr_ops:
            curr_ops.append(op)

    config.op_types = {str(k): v for k, v in enumerate(curr_ops)}

    for op in config.non_coreir_ops:
        if op not in config.op_types:
            config.op_types[op] = op

    config.op_types_flipped = {v: k for k, v in config.op_types.items()}

    subgraphs = utils.read_subgraphs(file_ind_pairs)

    utils.add_primitive_ops(subgraphs)

    for graph in subgraphs:
        utils.add_input_and_output_nodes(graph.subgraph)

    merger = DSEMerger(subgraphs)

    merger.merge_all_subgraphs()