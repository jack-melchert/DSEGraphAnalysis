import json
import sys
import os

def convert_dot(coreir_files):
    if not os.path.exists('.temp'):
        os.makedirs('.temp')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    op_types = {}
    instance_names = {}
    alu_supported_ops =  ["not", "and", "or", "xor", "shl", \
                        "lshr", "ashr", "neg", "add", "sub", \
                        "sle", "sge", "ule", "uge", "eq", "mux", \
                        "slt", "sgt", "ult", "ugt"]
    used_alu_ops = []

    files = []
    for f in coreir_files:
        files.append(open(f, 'r'))

    dot_files = [os.path.basename(f).replace(".json", ".dot") for f in coreir_files]

    instance_files = []
    connection_files = []

    for ind, file in enumerate(files):
        json_in = json.loads(file.read())
        json_insts = json_in['namespaces']['global']['modules']['DesignTop']['instances']
        json_connections = json_in['namespaces']['global']['modules']['DesignTop']['connections']
        new_insts = {}
        for k, v in json_insts.items():
            new_insts[k + '_' + str(ind)] = v

        new_connections = []
        for json_conn in json_connections:
            new_connections.append([json_conn[0].replace('.', '_' + str(ind) + '.'), json_conn[1].replace('.', '_' + str(ind) + '.')])
            
        instance_files.append(new_insts)
        connection_files.append(new_connections)

    op_index = 0

    for ind, instances in enumerate(instance_files):
        connections = connection_files[ind]

        with open('.temp/' + dot_files[ind], 'w') as out_file:
            
            inst_ind = 0
            out_file.write('t # 1\n')
            for inst_name, inst in instances.items():
                if 'genref' in inst and 'coreir' in inst['genref']:
                    op = inst['genref'].replace('coreir.', '')

                    if op in alu_supported_ops:
                        if not op in used_alu_ops:
                            used_alu_ops.append(op)

                    if op not in op_types:
                        op_types[op] = op_index
                        op_index += 1
                        
                    out_file.write('v ' + str(inst_ind) + ' ' + str(op_types[op]) + '\n')
                    instance_names[inst_name] = inst_ind
                    inst_ind += 1

            written_conn = []
            for conn in connections:
                if (conn[0].split('.', 1)[0] in instance_names and conn[1].split('.', 1)[0] in instance_names):
                    node0 = instance_names[conn[0].split('.', 1)[0]]
                    node1 = instance_names[conn[1].split('.', 1)[0]]

                    if "out" in conn[0].split('.', 1)[1]: 
                        out_file.write('e ' + str(node0) + ' ' + str(node1) + ' ' + conn[1].split('.', 1)[1].replace('in', '').replace('sel', '2') +'\n')
                    else:
                        out_file.write('e ' + str(node1) + ' ' + str(node0) + ' ' + conn[0].split('.', 1)[1].replace('in', '').replace('sel', '2') +'\n')


    with open('.temp/op_types.txt', 'w') as op_types_out_file:
        op_types_out_file.write(str(op_types))

    with open('.temp/used_ops.txt', 'w') as used_ops_out_file:
        used_ops_out_file.write(str(used_alu_ops))