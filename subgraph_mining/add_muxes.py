import json, sys
from peak_gen.arch import *

def add_mux_to_inputs(module):
    if not "in" in module["in0"]:
        if "in0" == module["in1"]:
            module["in0"] = [module["in0"], "in1"]
        else:
            module["in0"] = [module["in0"], "in0"]

    if not "in" in module["in1"]:
        if "in0" == module["in0"]:
            module["in1"] = [module["in1"], "in1"]
        else:
            module["in1"] = [module["in1"], "in0"]

def add_mux_to_output(module, outputs):

    if isinstance(outputs, list):
        for out_i in outputs:
            if isinstance(out_i, list):
                if module["id"] in out_i:
                    return
            else:
                if module["id"] in outputs:
                    return

        if not isinstance(outputs[0], list):
            outputs[0] = [outputs[0]]
        outputs[0].append(module["id"])

with open(str(sys.argv[1])) as json_file:
    json_in = json.loads(json_file.read())
    modules = json_in["modules"]
    outputs = json_in["outputs"]

    # First add mux to inputs of last alu
    for module in reversed(modules):
        if module["type"] == "alu":
            add_mux_to_inputs(module)
            break

    # Next add mux to output from first alu
    for module in modules:
        if module["type"] == "alu":
            add_mux_to_output(module, outputs)
            break


    # Then add mux to inputs of last mul
    for module in reversed(modules):
        if module["type"] == "mul":
            add_mux_to_inputs(module)
            break

    # Then add mux to output from first mul
    for module in modules:
        if module["type"] == "mul":
            add_mux_to_output(module, outputs)
            break


with open(str(sys.argv[1]).split(".")[0] + "_muxed.json", "w") as write_file:
    write_file.write(json.dumps(json_in, indent = 4, sort_keys=True))

print(json_in)
arch = read_arch(str(sys.argv[1]).split(".")[0] + "_muxed.json")
graph_arch(arch)