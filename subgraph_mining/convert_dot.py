import json
import sys
import os
import pickle

def convert_dot(coreir_files):



    if not os.path.exists('.temp'):
        os.makedirs('.temp')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    op_types = {}
    instance_names = {}
    supported_ops =  ["mul", "const", "not", "and", "or", "xor", "shl", \
                        "lshr", "ashr", "neg", "add", "sub", \
                        "sle", "sge", "ule", "uge", "eq", "mux", \
                        "slt", "sgt", "ult", "ugt", "smax", "smin", "umax", "umin", "absd", "abs"]
    used_ops = set()

    unused_ops = set()

    op_index = 0

    for ind, f in enumerate(coreir_files):
        
        print(f)

        file = open(f, 'r')
        json_in = json.loads(file.read())
        json_insts = json_in['namespaces']['global']['modules']['DesignTop']['instances']
        json_connections = json_in['namespaces']['global']['modules']['DesignTop']['connections']

        dot_files = [os.path.basename(f).replace(".json", ".dot") for f in coreir_files]


        with open('.temp/' + dot_files[ind], 'w') as out_file:
            
            inst_ind = 0
            out_file.write('t # 1\n')
            for name, inst in json_insts.items():
                if 'genref' in inst:
                    op = inst['genref'].split('.')[1]

                    if op in supported_ops:
                        used_ops.add(op)

                        if op not in op_types:
                            op_types[op] = op_index
                            op_index += 1

                        out_file.write('v ' + str(inst_ind) + ' ' + str(op_types[op]) + '\n')
                        instance_names[name] = inst_ind
                        inst_ind += 1
                    else:
                        unused_ops.add(op)


            for first, second in json_connections:
                first_name = first.split(".")[0]
                first_port = first.split(".")[1]
                second_name = second.split(".")[0]
                second_port = second.split(".")[1]
                if (first_name in instance_names and second_name in instance_names):
                    node0 = instance_names[first_name]
                    node1 = instance_names[second_name]

                    if first_port == "out":
                        out_file.write('e ' + str(node0) + ' ' + str(node1) + ' ' + second_port.replace('in', '').replace('sel', '2') +'\n')
                    else:
                        out_file.write('e ' + str(node1) + ' ' + str(node0) + ' ' + first_port.replace('in', '').replace('sel', '2') +'\n')


    with open('.temp/op_types.txt', 'wb') as op_types_out_file:
        # op_types_out_file.write(str(op_types))
        pickle.dump(op_types, op_types_out_file)

    with open('.temp/used_ops.txt', 'wb') as used_ops_out_file:
        # used_ops_out_file.write(str(used_ops))
        pickle.dump(used_ops, used_ops_out_file)


    print("Unused ops:")
    print(unused_ops)

