import argparse
import shutil
import os

from subgraph_mining.convert_dot import convert_coreir_to_dot
from subgraph_mining.graph_output import graph_output
from subgraph_mining.utils import *
from subgraph_merging.merge_subgraphs import merge_subgraphs
from subgraph_mining.find_maximal_ind_set import find_maximal_independent_set

def main():
    parser = argparse.ArgumentParser(description='Graph analysis of a coreir application')
    parser.add_argument('-c', '--cached', help='Use cached subgraphs', action="store_true", default=False)
    parser.add_argument('-f', '--files', nargs='+', metavar=("file", "subgraph_index"),help='Application files for analysis', action='append')
    parser.add_argument('-p', '--pipeline', help="Number of pipelining stages", type=int, default = 0)

    args = parser.parse_args()

    use_cached_subgraphs = args.cached

    file_ind_pairs = {}

    # breakpoint()
    for file_ind in args.files:
        if '.json' not in file_ind[0]:
            parser.error('-f file is not a json file')

        try:
            inds = [int(i) for i in file_ind[1:]]
        except:
            parser.error('subgraph indicies are not ints')

        file_ind_pairs[file_ind[0]] = inds

    file_names = list(file_ind_pairs.keys())

    # Cleaning output directories
    os.system("rm -rf .temp; rm -rf outputs; rm -rf pdf; rm -rf .ast_tools")
    # takes in .json coreIR file, outputs grami_in.txt and op_types.txt
    print("Starting .dot conversion...")
    convert_coreir_to_dot(file_names)
    print("Finished .dot conversion")

    dot_files = [".temp/" + os.path.basename(f).replace(".json", ".dot") for f in file_names]


    print("Starting application merging")
    merge_subgraphs(dot_files)

if __name__ == "__main__":
    main()
