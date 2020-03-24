import json
import sys
import os

op_types = {}
instance_names = {}

with open(str(sys.argv[1])) as json_file:
    json_in = json.loads(json_file.read())

    instances = json_in['namespaces']['global']['modules']['DesignTop']['instances']
    connections = json_in['namespaces']['global']['modules']['DesignTop']['connections']

    if not os.path.exists('.temp'):
        os.makedirs('.temp')

    with open('.temp/grami_in.txt', 'w') as out_file:
        
        out_file.write('t # 1\n')
        op_index = 0
        inst_ind = 0
        for inst_name, inst in instances.items():
            if 'genref' in inst:
                if inst['genref'] not in op_types:
                    op_types[inst['genref']] = op_index
                    op_index += 1
                out_file.write('v ' + str(inst_ind) + ' ' + str(op_types[inst["genref"]]) + '\n')
                instance_names[inst_name] = inst_ind
                inst_ind += 1

        written_conn = {}
        for conn in connections:
            if (conn[0].split('.', 1)[0] in instance_names and conn[1].split('.', 1)[0] in instance_names):
                if "out" in conn[0].split('.', 1)[1]:
                    if not instance_names[conn[0].split('.', 1)[0]] in written_conn or not written_conn[instance_names[conn[0].split('.', 1)[0]]] == instance_names[conn[1].split('.', 1)[0]]:
                        out_file.write('e ' + str(instance_names[conn[0].split('.', 1)[0]]) + ' ' + str(instance_names[conn[1].split('.', 1)[0]]) + ' 1\n')
                        written_conn[instance_names[conn[0].split('.', 1)[0]]] = instance_names[conn[1].split('.', 1)[0]]
                else:
                    if not instance_names[conn[1].split('.', 1)[0]] in written_conn or not written_conn[instance_names[conn[1].split('.', 1)[0]]] == instance_names[conn[0].split('.', 1)[0]]:
                        out_file.write('e ' + str(instance_names[conn[1].split('.', 1)[0]]) + ' ' + str(instance_names[conn[0].split('.', 1)[0]]) + ' 1\n')
                        written_conn[instance_names[conn[1].split('.', 1)[0]]] = instance_names[conn[0].split('.', 1)[0]]

with open('.temp/op_types.txt', 'w') as op_types_out_file:
    op_types_out_file.write(str(op_types))