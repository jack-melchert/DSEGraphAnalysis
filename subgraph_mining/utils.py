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

    max_subgraph = max(subgraph_inds)

    # TEMPORARY to speed up subgraph mining
    support_dict = {}
    support_dict[".temp/camera_pipeline.dot"] = "20"
    support_dict[".temp/conv_3_3.dot"] = "15"
    support_dict[".temp/harris.dot"] = "15"
    support_dict[".temp/strided_conv.dot"] = "15"

    # if input_file not in support_dict:
    support = 30 # Starting support number
    # else:
    #     support = support_dict[input_file]

    print("Starting GraMi subgraph mining...")
    
    num_subgraphs = 0

    while num_subgraphs <= max_subgraph:
        
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
