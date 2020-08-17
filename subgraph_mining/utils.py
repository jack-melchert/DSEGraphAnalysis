import os 

def reverse_subgraph_list(input_filename, output_filename):

    with open(input_filename) as file:
        lines = file.readlines()

    graph_num = -1

    out_text = []

    for line in lines:
        if ':' in line:
            graph_num += 1
            out_text.append(line)
        elif graph_num >= 0:
            out_text[graph_num] += line

    out_text.reverse()

    if not os.path.exists('.temp'):
        os.makedirs('.temp')

    with open(output_filename, "w") as outfile:
        for subgraph in out_text:
            outfile.write(subgraph)


def grami_subgraph_mining(input_file, subgraph_inds):

    if len(subgraph_inds) != 0:
        max_subgraph = max(subgraph_inds)
        support = 13 # Starting support number

        print("Starting GraMi subgraph mining...")
        
        num_subgraphs = 0

        while num_subgraphs <= max_subgraph and support > 0:
            
            os.system('''cd GraMi
            ./grami -f ../../''' + input_file + ''' -s ''' + str(support) + ''' -t 1 -p 0 > grami_log.txt
            cd ../''')

            with open("GraMi/Output.txt") as file:
                lines = file.readlines()

            num_subgraphs = 0
            
            for line in lines:
                if ':' in line:
                    num_subgraphs += 1

            print("Support =", support, " num_subgraphs =", num_subgraphs)

            support -= 1

        print("Finished GraMi subgraph mining")
