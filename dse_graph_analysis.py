import argparse
import shutil
import os

from subgraph_mining.convert_dot import convert_dot
from subgraph_mining.graph_output import graph_output
from subgraph_mining.convert_subgraphs_to_arch import convert_subgraphs_to_arch
from subgraph_mining.utils import *
from subgraph_merging.merge_subgraphs import merge_subgraphs


def main():
    parser = argparse.ArgumentParser(description='Graph analysis of a coreir application')
    parser.add_argument('-c', '--cached', help='Use cached subgraphs', action="store_true", default=False)
    parser.add_argument('-f', '--files', nargs='+', metavar=("file", "subgraph_index"),help='Application files for analysis', action='append')


    args = parser.parse_args()

    use_cached_subgraphs = args.cached

    file_ind_pairs = {}

    for file_ind in args.files:
        if '.json' not in file_ind[0]:
            parser.error('-f file is not a json file')

        try:
            inds = [int(i) for i in file_ind[1:]]
        except:
            parser.error('subgraph indicies are not ints')

        file_ind_pairs[file_ind[0]] = inds

    file_names = list(file_ind_pairs.keys())
    subgraph_file_ind_pairs = {}



    if not use_cached_subgraphs:
        # Cleaning output directories
        os.system("rm -rf .temp; rm -rf outputs; rm -rf pdf; rm -rf .ast_tools")
        # takes in .json coreIR file, outputs grami_in.txt and op_types.txt
        print("Starting .dot conversion...")
        convert_dot(file_names)
        print("Finished .dot conversion")

        dot_files = [".temp/" + os.path.basename(f).replace(".json", ".dot") for f in file_names]

        for file_ind, file in enumerate(file_names):

            file_stripped = os.path.basename(file).split(".")[0]

            print("Starting on ", file_stripped)

            # Takes in grami_in.txt produces orig_graph.pdf
            print("Graphing original graph")
            graph_output(dot_files[file_ind], file_stripped)

            # Takes in grami_in.txt and subgraph support, produces Output.txt
            grami_subgraph_mining(dot_files[file_ind], file_ind_pairs[file])


            new_subgraphs_file = dot_files[file_ind].replace(".dot", "_subgraphs.dot")
            reverse_subgraph_list("GraMi/Output.txt", new_subgraphs_file)

            # Takes in Output.txt produces subgraphs.pdf
            print("Graphing subgraphs")
            graph_output(new_subgraphs_file, file_stripped + "_subgraphs")

            subgraph_file_ind_pairs[dot_files[file_ind].replace(".dot", "_subgraphs.dot")] = file_ind_pairs[file]
    else:
        dot_files = [".temp/" + os.path.basename(f).replace(".json", ".dot") for f in file_names]
        for file_ind, file in enumerate(file_names):
            subgraph_file_ind_pairs[dot_files[file_ind].replace(".dot", "_subgraphs.dot")] = file_ind_pairs[file]

    print("Starting subgraph merging")
    merge_subgraphs(subgraph_file_ind_pairs)

if __name__ == "__main__":
    main()