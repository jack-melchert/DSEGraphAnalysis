import json
import coreir
import sys
import os
import pickle
import subgraph_merging.config as config
from graphviz import Digraph

def convert_coreir_to_dot(coreir_files):

    if not os.path.exists('.temp'):
        os.makedirs('.temp')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    stripped_files = [os.path.basename(f).replace(".json", "") for f in coreir_files]
    dot_files = [os.path.basename(f).replace(".json", ".dot") for f in coreir_files]

    op_types = {}
    op_types["self"] = "0"
    instance_names = {}
    used_ops = set()
    unsupported_ops = set()

    op_index = 1

    op = "bitconst"
    used_ops.add("bitconst")
    op_types[op] = op_index
    op_index += 1
    op = "sge"
    used_ops.add("sge")
    op_types[op] = op_index
    op_index += 1
   
    for ind, f in enumerate(coreir_files):
        
        print(f)
        out_file = open('.temp/' + dot_files[ind], 'w')

        inst_ind = 1
        instance_names["self"] = 0
        out_file.write('t # 1\n')
        out_file.write('v ' + str(0) + ' ' + '0' + '\n')

        c = coreir.Context()
        c.load_library("commonlib")
        cmod = c.load_from_file(f)
        c.run_passes(["rungenerators", "flattentypes", "flatten", "deletedeadinstances"])
        kernels = dict(c.global_namespace.modules)
        dot = Digraph()
        wired_self = False

        for kname, kmod in kernels.items():
            for inst in kmod.definition.instances:
                namespace = inst.module.namespace.name
                op = inst.module.name
                if namespace == 'corebit':
                    op = f"bit{op}"
                dot.node(inst.name, op)
                if op in config.supported_ops:
                    used_ops.add(op)
                    if op not in op_types:
                        op_types[op] = op_index
                        op_index += 1
                    out_file.write('v ' + str(inst_ind) + ' ' + str(op_types[op]) + '\n')
                    instance_names[inst.name] = inst_ind
                    inst_ind += 1
                else:
                    unsupported_ops.add(op)

        for kname, kmod in kernels.items():
            for conn in kmod.definition.connections:

                if conn.first.type.is_input():
                    assert(conn.second.type.is_output())
                    source = conn.second.selectpath
                    sink = conn.first.selectpath
                else:
                    assert(conn.first.type.is_output())
                    assert(conn.second.type.is_input())
                    source = conn.first.selectpath
                    sink = conn.second.selectpath

                # if source[0] == "self" and not wired_self:
                #     wired_self = True
                #     dot.edge(source[0], sink[0])

                if source[0] != "self":
                    dot.edge(source[0], sink[0])
                    if source[0] in instance_names and sink[0] in instance_names:
                        if sink[0] != "self":
                            out_file.write('e ' + str(instance_names[source[0]]) + ' ' + str(instance_names[sink[0]]) + ' ' + sink[1].replace('in', '').replace('sel', '2') +'\n')
                        else:
                            out_file.write('e ' + str(instance_names[source[0]]) + ' ' + str(instance_names[sink[0]]) + ' 0' +'\n')
                    

    with open('.temp/op_types.txt', 'wb') as op_types_out_file:
        # op_types_out_file.write(str(op_types))
        pickle.dump(op_types, op_types_out_file)

    with open('.temp/used_ops.txt', 'wb') as used_ops_out_file:
        # used_ops_out_file.write(str(used_ops))
        pickle.dump(used_ops, used_ops_out_file)

    dot.render(f'pdf/{dot_files[ind]}', view=False)  
    print("Used ops:", used_ops)
    print("Unsupported ops:", unsupported_ops)
