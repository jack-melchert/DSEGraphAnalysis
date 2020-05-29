import argparse


def main():
    parser = argparse.ArgumentParser(description='Graph analysis of a coreir application')

    parser.add_argument('-f', '--files', nargs='+', metavar=("file", "subgraph_index"),help='Application files for analysis', action='append')


    args = parser.parse_args()

    file_ind_pairs = {}

    for file_ind in args.files:
        if '.json' not in file_ind[0]:
            parser.error('-f file is not a json file')

        try:
            inds = [int(i) for i in file_ind[1:]]
        except:
            parser.error('subgraph indicies are not ints')

        file_ind_pairs[file_ind[0]] = inds

    print(file_ind_pairs)

if __name__ == "__main__":
    main()