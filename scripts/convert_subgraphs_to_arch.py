import sys
import ast
import json
import os

def sort_modules(modules):
    ids = []

    output_modules = []
    while len(modules) > 0:
        for module in modules:
            if module['type'] == 'const':
                ids.append(module["id"])
                output_modules.append(module)
                modules.remove(module)

            if "in0" in module:
                if module["in0"] in ids:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)

            if "in1" in module:
                if module["in1"] in ids:
                    if module in modules:
                        ids.append(module["id"])
                        output_modules.append(module)
                        modules.remove(module)

            if "in0" in module and "in1" in module:
                if "in" in module["in0"] and "in" in module["in1"]:
                    ids.append(module["id"])
                    output_modules.append(module)
                    modules.remove(module)

    return output_modules

with open(str(sys.argv[1])) as file:
    lines = file.readlines()

with open(".temp/op_types.txt") as file:
    op_types = ast.literal_eval(file.read())

op_types = {str(v): k for k, v in op_types.items()}

arches = []
modules = {}
ids = []
connected_ids = []

for line in lines:
    if ':' in line or '#' in line:
        if len(modules) > 0:
            arch["modules"] = [v for v in modules.values()]
            arch["outputs"] = [i for i in ids if i not in connected_ids]
            arches.append(arch)
        arch = {}
        arch["input_width"] = 16
        arch["output_width"] = 16
        arch["enable_input_regs"] = False
        arch["enable_output_regs"] = False
        modules = {}
        ids = []
        connected_ids = []
        
    elif 'v' in line:
        # Node id : line.split()[1], type : op_types[line.split()[2]]
        modules[line.split()[1]] = {}
        modules[line.split()[1]]["id"] = line.split()[1]
        modules[line.split()[1]]["type"] = op_types[line.split()[2]].replace('coreir.','').replace('add','alu')
        ids.append(line.split()[1])
    elif 'e' in line:
        # import pdb; pdb.set_trace()
        # Edge from line.split()[1] to line.split()[2]
        connected_ids.append(line.split()[1])
        if 'in0' in modules[line.split()[2]]:
            modules[line.split()[2]]["in1"] = line.split()[1]
        else:
            modules[line.split()[2]]["in0"] = line.split()[1]
    

arch["modules"] = [v for v in modules.values()]
arch["outputs"] = [i for i in ids if i not in connected_ids]
arches.append(arch)
arch = {}
arch["input_width"] = 16
arch["output_width"] = 16
arch["enable_input_regs"] = False
arch["enable_output_regs"] = False

if not os.path.exists('outputs'):
    os.makedirs('outputs')

for sub_ind, arch_out in enumerate(arches):
    input_counter = 0
    for module in arch_out["modules"]:
        if not module['type'] == 'const':
            if 'in0' not in module:
                module["in0"] = "in" + str(input_counter)
                input_counter += 1
            if 'in1' not in module:
                module["in1"] = "in" + str(input_counter)
                input_counter += 1

    arch_out["modules"] = sort_modules(arch_out["modules"])

    with open("outputs/subgraph_arch_" + str(sub_ind) + ".json", "w") as write_file:
        write_file.write(json.dumps(arch_out, indent = 4, sort_keys=True))
