import json
import sys
import os

if not os.path.exists('.temp'):
    os.makedirs('.temp')
if not os.path.exists('outputs'):
    os.makedirs('outputs')
op_types = {}
instance_names = {}
alu_supported_ops =  ["not", "and", "or", "xor", "shl", "lshr", "ashr", "neg", "add", "sub", "sle", "sge", "ule", "uge", "eq", "mux", "slt", "sgt", "ult", "ugt"]
used_alu_ops = []

files = []
files_still = True
idx = 1
while files_still:
    try:
        files.append(open(str(sys.argv[idx]), "r"))
        idx += 1
    except:
        files_still = False


instances = {}
connections = []

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
        

    instances.update(new_insts)
    connections += new_connections



with open('.temp/grami_in.txt', 'w') as out_file:
    
    out_file.write('t # 1\n')
    op_index = 0
    inst_ind = 0
    for inst_name, inst in instances.items():
        if 'genref' in inst:
            op = inst['genref'].replace('coreir.', '')

            if op in alu_supported_ops:
                if not op in used_alu_ops:
                    used_alu_ops.append(op)
                op = "alu"

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
                if not [node0, node1] in written_conn:
                    out_file.write('e ' + str(node0) + ' ' + str(node1) + ' 1\n')
                    written_conn.append([node0, node1])
            else:
                if not [node1, node0] in written_conn:
                    out_file.write('e ' + str(node1) + ' ' + str(node0) + ' 1\n')
                    written_conn.append([node1, node0])



with open('.temp/op_types.txt', 'w') as op_types_out_file:
    op_types_out_file.write(str(op_types))

with open('outputs/used_ops.txt', 'w') as used_ops_out_file:
    used_ops_out_file.write(str(used_alu_ops))