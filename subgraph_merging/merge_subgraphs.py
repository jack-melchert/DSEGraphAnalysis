import pickle

import subgraph_merging.utils as utils
from .merger import DSEMerger 
import subgraph_merging.config as config

def merge_subgraphs(file_ind_pairs, pipeline):
    utils.clean_output_dirs()

    with open(".temp/op_types.txt", "rb") as file:
        op_types_from_file = pickle.load(file)

    curr_ops = [*op_types_from_file]

    for op in config.primitive_ops:
        if op not in curr_ops:
            curr_ops.append(op)
    
    config.op_types = {str(k): v for k, v in enumerate(curr_ops)}

    for op in config.non_coreir_ops:
        if op not in config.op_types.values():
            config.op_types[op] = op

    config.op_types_flipped = {v: k for k, v in config.op_types.items()}

    print("Reading subgraphs")

    subgraphs = utils.read_subgraphs(file_ind_pairs)

    utils.add_primitive_ops(subgraphs)

    print("Generating peak_eqs")

    for sub_idx, graph in enumerate(subgraphs):
        graph.add_input_and_output_nodes()

    print("Merging subgraphs")
    merger = DSEMerger(subgraphs)
    merger.merge_all_subgraphs()

    merger.merged_graph_to_arch()
    merger.write_merged_graph_arch()

    mul_ops = []

    for sub_idx, graph in enumerate(subgraphs):
        graph.generate_peak_eq()
        graph.write_peak_eq("outputs/peak_eqs/peak_eq_" + str(sub_idx) + ".py")
        mul_ops.append(graph.contains_mul())



    print("Generating rewrite rules")
    merger.generate_rewrite_rules(mul_ops)
    merger.write_rewrite_rules()


    if pipeline > 0:
        merger.merged_graph.pipeline(pipeline)

    print("Translating to arch")
    merger.merged_graph_to_arch()
    merger.write_merged_graph_arch()
    merger.merged_graph.analyze_pe()
    merger.print_area_and_energy()
