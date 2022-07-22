import argparse
import shutil
import os
import glob

from subgraph_mining.convert_dot import convert_coreir_to_dot
from subgraph_mining.graph_output import graph_output
from subgraph_mining.utils import *
from subgraph_merging.merge_subgraphs import merge_subgraphs
from subgraph_mining.find_maximal_ind_set import find_maximal_independent_set

def main():
    parser = argparse.ArgumentParser(description='Graph analysis of a coreir application')
    parser.add_argument('-c', '--cached', help='Use cached subgraphs', action="store_true", default=False)
    parser.add_argument('-d', '--directory', type=str, default="", dest="directory", help='Application directory with files for analysis')
    parser.add_argument('-s', '--subgraphs', type=int, default=5, help='Num subgraphs to display')

    args = parser.parse_args()

    use_cached_subgraphs = args.cached

    file_names = args.directory


    if not use_cached_subgraphs:
        # Cleaning output directories
        os.system("rm -rf .temp; rm -rf outputs; rm -rf pdf; rm -rf .ast_tools")
        # takes in .json coreIR file, outputs grami_in.txt and op_types.txt
        print("Starting .dot conversion...")
        convert_coreir_to_dot(file_names)
        print("Finished .dot conversion")

        # dot_files = [".temp/" + os.path.basename(f).replace(".json", ".dot") for f in file_names]

        # for file_ind, file in enumerate(file_names):
        file_ind = 0
        dot_files = [".temp/combined_graph.dot"]
        file = ".temp/combined_graph.dot"

        # Takes in grami_in.txt and subgraph support, produces Output.txt
        grami_subgraph_mining(dot_files[file_ind], int(args.subgraphs))
        
        max_ind_set_stats = find_maximal_independent_set(dot_files[file_ind], "GraMi/Output.txt")

        new_subgraphs_file = dot_files[file_ind].replace(".dot", "_subgraphs.dot")
        max_ind_set_stats = sort_subgraph_list("GraMi/Output.txt", new_subgraphs_file, max_ind_set_stats)

        # Takes in Output.txt produces subgraphs.pdf
        print("Graphing subgraphs")
        graph_output(new_subgraphs_file, "combined_graph_subgraphs", max_ind_set_stats)



if __name__ == "__main__":
    main()
