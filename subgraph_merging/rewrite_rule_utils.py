import importlib
import time
import os
import json
import peak
import magma as m

from hwtypes import BitVector, Tuple, Bit

from peak import family
from peak.mapper import ArchMapper
from peak.mapper import RewriteRule
from peak.mapper.utils import pretty_print_binding
from peak.assembler.assembled_adt import  AssembledADT
from peak.assembler.assembler import Assembler


from peak_gen.peak_wrapper import wrapped_peak_class
from peak_gen.arch import read_arch, graph_arch
from peak_gen.isa import inst_arch_closure, ALU_t, Signed_t, MUL_t, BIT_ALU_t
from peak_gen.asm import asm_arch_closure
from peak_gen.cond import Cond_t

import subgraph_merging.config as config


def gen_rewrite_rule(node_dict):
    rr = {}
    for k, v in node_dict.items():
        rr[k] = {}
        rr[k]['0'] = v['0']
        if not (v['alu_op'] == "const" or v['alu_op'] == "bitconst" or v['alu_op'] == "output" or v['alu_op'] == "bit_output"):
            rr[k]['1'] = v['1']

        if v['alu_op'] == "mux" or v['alu_op'] in config.lut_supported_ops:
            rr[k]['2'] = v['2']

        rr[k]['alu_op'] = v['alu_op']

    print(rr)
    return rr


def formulate_rewrite_rules(rrules, merged_arch):


    op_map = {}

    op_map["Mult0"] = MUL_t.Mult0
    op_map["Mult1"] = MUL_t.Mult1
    # op_map["not"] = ALU_t.Sub  
    op_map["and"] = BIT_ALU_t.And    
    op_map["or"] = BIT_ALU_t.Or    
    op_map["xor"] = BIT_ALU_t.XOr    
    op_map["shl"] = ALU_t.SHL   
    op_map["lshr"] = ALU_t.SHR    
    op_map["ashr"] = ALU_t.SHR    
    op_map["neg"] = ALU_t.Sub   
    op_map["add"] = ALU_t.Add    
    op_map["sub"] = ALU_t.Sub    
    op_map["sle"] = ALU_t.LTE_Min    
    op_map["sge"] = ALU_t.GTE_Max     
    op_map["slt"] = ALU_t.Sub   
    op_map["sgt"] = ALU_t.Sub  
    op_map["uge"] = ALU_t.GTE_Max    
    op_map["ule"] = ALU_t.LTE_Min   
    op_map["ugt"] = ALU_t.Sub    
    op_map["ult"] = ALU_t.Sub    
    op_map["eq"] = ALU_t.Sub    
    op_map["umax"] = ALU_t.GTE_Max    
    op_map["smax"] = ALU_t.GTE_Max  
    op_map["umin"] = ALU_t.LTE_Min  
    op_map["smin"] = ALU_t.LTE_Min
    op_map["abs"] = ALU_t.Abs
    op_map["absd"] = ALU_t.Absd
    op_map["bitand"] = BitVector[8](136)
    op_map["bitor"] = BitVector[8](252)
    op_map["bitxor"] = BitVector[8](90)
    op_map["bitnot"] = BitVector[8](85)
    op_map["bitmux"] = BitVector[8](228)
    op_map["bitconst"] = ALU_t.And



    cond_map = {}
    cond_map["sub"] = Cond_t.ALU    
    cond_map["sle"] = Cond_t.ALU    
    cond_map["sge"] = Cond_t.ALU     
    cond_map["slt"] = Cond_t.SLT   
    cond_map["sgt"] = Cond_t.SGT  
    cond_map["ult"] = Cond_t.ULT    
    cond_map["ugt"] = Cond_t.UGT    
    cond_map["ule"] = Cond_t.ALU   
    cond_map["uge"] = Cond_t.ALU    
    cond_map["eq"] = Cond_t.EQ
    cond_map["gte"] = Cond_t.ALU
    cond_map["lte"] = Cond_t.ALU
    # cond_map["Mult0"] = Cond_t.ALU
    # cond_map["Mult1"] = Cond_t.ALU
    # cond_map["not"] = Cond_t.ALU  
    # cond_map["and"] = Cond_t.ALU    
    # cond_map["or"] = Cond_t.ALU    
    # cond_map["xor"] = Cond_t.ALU    
    # cond_map["shl"] = Cond_t.ALU   
    # cond_map["lshr"] = Cond_t.ALU    
    # cond_map["ashr"] = Cond_t.ALU    
    # cond_map["neg"] = Cond_t.ALU   
    # cond_map["add"] = Cond_t.ALU    
    cond_map["umax"] = Cond_t.ALU    
    cond_map["smax"] = Cond_t.ALU  
    cond_map["umin"] = Cond_t.ALU  
    cond_map["smin"] = Cond_t.ALU
    # cond_map["abs"] = Cond_t.ALU
    # cond_map["absd"] = Cond_t.ALU
    # cond_map["bitand"] = Cond_t.ALU
    # cond_map["bitor"] = Cond_t.ALU
    # cond_map["bitxor"] = Cond_t.ALU
    # cond_map["bitnot"] = Cond_t.ALU
    # cond_map["bitmux"] = Cond_t.ALU
    # cond_map["bitconst"] = Cond_t.ALU

    # signed_map = {}
    # signed_map["Mult0"] = Signed_t.unsigned
    # signed_map["Mult1"] = Signed_t.unsigned
    # # signed_map["not"] = Signed_t.unsigned  
    # signed_map["and"] = Signed_t.unsigned    
    # signed_map["or"] = Signed_t.unsigned    
    # signed_map["xor"] = Signed_t.unsigned    
    # signed_map["shl"] = Signed_t.unsigned   
    # signed_map["lshr"] = Signed_t.unsigned    
    # signed_map["ashr"] = Signed_t.signed    
    # signed_map["neg"] = Signed_t.unsigned   
    # signed_map["add"] = Signed_t.unsigned    
    # signed_map["sub"] = Signed_t.unsigned    
    # signed_map["sle"] = Signed_t.signed    
    # signed_map["sge"] = Signed_t.signed     
    # signed_map["slt"] = Signed_t.signed   
    # signed_map["sgt"] = Signed_t.signed  
    # signed_map["ult"] = Signed_t.unsigned    
    # signed_map["ugt"] = Signed_t.unsigned    
    # signed_map["ule"] = Signed_t.unsigned   
    # signed_map["uge"] = Signed_t.unsigned    
    # signed_map["eq"] = Signed_t.unsigned
    # signed_map["umax"] = Signed_t.unsigned    
    # signed_map["smax"] = Signed_t.signed  
    # signed_map["umin"] = Signed_t.unsigned  
    # signed_map["smin"] = Signed_t.signed
    # signed_map["abs"] = Signed_t.unsigned
    # signed_map["absd"] = Signed_t.unsigned


    signed_map = {}
    signed_map["Mult0"] = Signed_t.unsigned
    signed_map["mul"] = Signed_t.unsigned
    signed_map["Mult1"] = Signed_t.unsigned
    # signed_map["ult"] = Signed_t.unsigned    
    # signed_map["ugt"] = Signed_t.unsigned    
    signed_map["ule"] = Signed_t.unsigned   
    signed_map["uge"] = Signed_t.unsigned    
    signed_map["sle"] = Signed_t.signed    
    signed_map["sge"] = Signed_t.signed     
    # signed_map["slt"] = Signed_t.signed   
    # signed_map["sgt"] = Signed_t.signed  
    signed_map["lshr"] = Signed_t.unsigned    
    signed_map["ashr"] = Signed_t.signed    
    signed_map["shr"] = Signed_t.signed  
    signed_map["umax"] = Signed_t.unsigned    
    signed_map["smax"] = Signed_t.signed  
    signed_map["umin"] = Signed_t.unsigned  
    signed_map["smin"] = Signed_t.signed
    signed_map["gte"] = Signed_t.unsigned
    signed_map["lte"] = Signed_t.unsigned
    # signed_map["sub"] = Signed_t.unsigned
    # # signed_map["not"] = Signed_t.unsigned  
    # signed_map["and"] = Signed_t.unsigned    
    # signed_map["or"] = Signed_t.unsigned    
    # signed_map["xor"] = Signed_t.unsigned    
    # signed_map["shl"] = Signed_t.unsigned   
    # signed_map["neg"] = Signed_t.unsigned   
    # signed_map["add"] = Signed_t.unsigned    
    # signed_map["sub"] = Signed_t.unsigned    
    # signed_map["eq"] = Signed_t.unsigned
    # signed_map["abs"] = Signed_t.unsigned
    # signed_map["absd"] = Signed_t.unsigned

    rrules_out = []

    arch = read_arch("./outputs/PE.json")
    graph_arch(arch)
    PE_fc = wrapped_peak_class(arch)
    arch_mapper = ArchMapper(PE_fc)

    for sub_idx, rrule in enumerate(rrules):
        rr_output = {}
      
        input_mappings = {}
        bit_input_mappings = {}
        const_mappings = {}
        bitconst_mappings = {}
        seen_inputs = []

        alu = []
        bit_alu = []
        mul = []
        lut = []
        ops = []
        cond = []
        signed = []
        mux_in0 = []
        mux_in1 = []
        mux_in2 = []
        mux_out = []
        mux_bit_out = []
        cfg_idx = 0
        

        for module in merged_arch["modules"]:
            if module["id"] in rrule:
                v = rrule[module["id"]]

                if module["type"] == "const" or module["type"] == "bitconst":
                    # cfg_idx = int(v['0'].split("const")[1])
                    const_mappings[cfg_idx] = v['0']
                    cfg_idx += 1
                else: #if module["type"] == "alu" or module["type"] == "bit_alu" or module["type"] == "lut" or module["type"] == "mul" or module["type"] == "mux":
                    if module["type"] == "alu":
                        alu.append(v['alu_op'])
                        ops.append(v['alu_op'])
                    elif module["type"] == "bit_alu":
                        bit_alu.append(v['alu_op'])
                        ops.append(v['alu_op'])
                    elif module["type"] == "lut":
                        lut.append(v['alu_op'])
                    elif module["type"] == "mul":
                        mul.append("Mult0")
                        ops.append("Mult0")
                    else:
                        ops.append(v['alu_op'])

                    if module["type"] in cond_map:
                        cond.append(v['alu_op'])

                    if module["type"] in signed_map:
                        signed.append(v['alu_op'])



                    if len(module["in0"]) > 1:
                        for mux_idx, mux_in in enumerate(module["in0"]):
                            if v['0'] == mux_in:
                                mux_in0.append(mux_idx)

                                if "in" in v['0']:
                                    in_idx = int(mux_in.split("in")[1])
                                    if module["type"] == "lut":
                                        bit_input_mappings[in_idx] = v['0']
                                    else:
                                        input_mappings[in_idx] = v['0']
                                    seen_inputs.append(v['0'])
                            elif "in" in mux_in:
                                in_idx = int(mux_in.split("in")[1])

                                if mux_in not in seen_inputs:
                                    if module["type"] == "lut":
                                        bit_input_mappings[in_idx] = 0
                                    else:
                                        input_mappings[in_idx] = 0
                                    seen_inputs.append(mux_in)

                    elif "in" in module["in0"][0]:
                        # try:
                        in_idx = int(v['0'].split("in")[1])
                        # except:
                            # breakpoint()
                        if module["type"] == "lut":
                            bit_input_mappings[in_idx] = v['0']
                        else:
                            input_mappings[in_idx] = v['0']
                        seen_inputs.append(v['0'])

                    # if '1' in v:
                    if len(module["in1"]) > 1:
                        for mux_idx, mux_in in enumerate(module["in1"]):
                            if v['1'] == mux_in:
                                mux_in1.append(mux_idx)

                                if "in" in v['1']:
                                    in_idx = int(mux_in.split("in")[1])
                                    if module["type"] == "lut":
                                        bit_input_mappings[in_idx] = v['1']
                                    else:
                                        input_mappings[in_idx] = v['1']
                                    seen_inputs.append(v['1'])
                            elif "in" in mux_in:
                                in_idx = int(mux_in.split("in")[1])

                                if mux_in not in seen_inputs:
                                    if module["type"] == "lut":
                                        bit_input_mappings[in_idx] = 0
                                    else:
                                        input_mappings[in_idx] = 0
                                    seen_inputs.append(mux_in)

                    elif "in" in module["in1"][0]:
                        in_idx = int(v['1'].split("in")[1])
                        if module["type"] == "lut":
                            bit_input_mappings[in_idx] = v['1']
                        else:
                            input_mappings[in_idx] = v['1']
                        seen_inputs.append(v['1'])

                    if '2' in v:
                        if len(module["in2"]) > 1:
                            for mux_idx, mux_in in enumerate(module["in2"]):
                                if v['2'] == mux_in:
                                    mux_in2.append(mux_idx)

                                    if "in" in v['2']:
                                        in_idx = int(mux_in.split("in")[1])

                                        if "mux" in v['alu_op']:
                                            bit_input_mappings[in_idx] = v['2']
                                        else:
                                            bit_input_mappings[in_idx] = 0
                                        seen_inputs.append(v['2'])
                                elif "in" in mux_in:
                                    in_idx = int(mux_in.split("in")[1])

                                    if mux_in not in seen_inputs:
                                        bit_input_mappings[in_idx] = 0
                                        seen_inputs.append(mux_in)

                        elif "in" in module["in2"][0]:
                            in_idx = int(v['2'].split("in")[1])
                            if "mux" in v['alu_op']:
                                bit_input_mappings[in_idx] = v['2']
                            else:
                                bit_input_mappings[in_idx] = 0
                            seen_inputs.append(v['2'])


            else:
                if module["type"] == "const" or module["type"] == "bitconst":
                    const_mappings[cfg_idx] = 0
                    cfg_idx += 1
                elif module["type"] == "lut":
                    lut.append("bitand")
                    

                    if len(module["in0"]) > 1:
                        mux_in0.append(0)

                    if len(module["in1"]) > 1:
                        mux_in1.append(0)

                    if len(module["in2"]) > 1:
                        mux_in2.append(0)

                    for mux_in in module["in0"]:
                        if "in" in mux_in:
                            in_idx = int(mux_in.split("in")[1])

                            if mux_in not in seen_inputs:
                                bit_input_mappings[in_idx] = 0
                                seen_inputs.append(mux_in)

                    for mux_in in module["in1"]:
                        if "in" in mux_in:
                            in_idx = int(mux_in.split("in")[1])

                            if mux_in not in seen_inputs:
                                bit_input_mappings[in_idx] = 0
                                seen_inputs.append(mux_in)

                    for mux_in in module["in2"]:
                        if "in" in mux_in:
                            in_idx = int(mux_in.split("in")[1])

                            if mux_in not in seen_inputs:
                                bit_input_mappings[in_idx] = 0
                                seen_inputs.append(mux_in)
                else: #if module["type"] == "alu" or module["type"] == "mul" or module["type"] == "mux":
                    if module["type"] == "alu":
                        alu.append("add")
                        ops.append("add")
                    elif module["type"] == "bit_alu":
                        bit_alu.append("and")
                        ops.append("and")
                    elif module["type"] == "mul":
                        mul.append("Mult0") 
                        ops.append("Mult0") 
                    elif module["type"] == "mux":
                        if len(module["in2"]) > 1:
                            mux_in2.append(0)
                        for mux_in in module["in2"]:
                            if "in" in mux_in:
                                in_idx = int(mux_in.split("in")[1])

                                if mux_in not in seen_inputs:
                                    bit_input_mappings[in_idx] = 0
                                    seen_inputs.append(mux_in)
                    else:
                        ops.append("add")


                    if module["type"] in cond_map:
                        cond.append("sub")

                    if module["type"] in signed_map:
                        signed.append("ule")

                    if len(module["in0"]) > 1:
                        mux_in0.append(0)

                    if len(module["in1"]) > 1:
                        mux_in1.append(0)

                    for mux_in in module["in0"]:
                        if "in" in mux_in:
                            in_idx = int(mux_in.split("in")[1])

                            if mux_in not in seen_inputs:
                                input_mappings[in_idx] = 0
                                seen_inputs.append(mux_in)

                    for mux_in in module["in1"]:
                        if "in" in mux_in:
                            in_idx = int(mux_in.split("in")[1])

                            if mux_in not in seen_inputs:
                                input_mappings[in_idx] = 0
                                seen_inputs.append(mux_in)
            

        if len(arch.outputs) > 0 and len(arch.outputs[0]) > 1:
            if "out0" in rrule:
                output_id = rrule["out0"]["0"]
            else:
                output_id = merged_arch["outputs"][0][0]
            mux_out.append(arch.outputs[0].index(output_id))


        if len(arch.bit_outputs) > 0 and len(arch.bit_outputs[0]) > 1:
            if "bit_out0" in rrule:
                bit_output_id = rrule["bit_out0"]["0"]
            else:
                bit_output_id = merged_arch["bit_outputs"][0][0]
            mux_bit_out.append(arch.bit_outputs[0].index(bit_output_id))

        input_binding = []

        input_idx = 0
        for k, v in input_mappings.items():
            if v == 0:
                input_binding.append((peak.mapper.utils.Unbound, ("inputs"+str(k), )))
            else:
                input_binding.append(((v,), ("inputs"+str(k), )))
            input_idx += 1

        for k, v in bit_input_mappings.items():
            if v == 0:
                input_binding.append((peak.mapper.utils.Unbound, ("inputs"+str(k+input_idx), )))
            else:
                input_binding.append(((v,), ("inputs"+str(k+input_idx), )))


        for k, v in const_mappings.items():
            if v == 0:
                input_binding.append((BitVector[16](0), ("inst", "const_data", k)))
            else:
                input_binding.append(((v,), ("inst", "const_data", k)))

        # for k, v in bitconst_mappings.items():
        #     if v == 0:
        #         input_binding.append((peak.mapper.utils.Unbound, ("inst", "const_data", k)))
        #     else:
        #         input_binding.append(((v,), ("inst", "const_data", k)))




        # print(signed)
        signed = [signed_map[n] for n in signed]
        cond = [cond_map[n] for n in cond]
        alu = [op_map[n] for n in alu]
        bit_alu = [op_map[n] for n in bit_alu]
        mul = [op_map[n] for n in mul]
        lut = [op_map[n] for n in lut]

        mux_in0_bw = [m.math.log2_ceil(len(arch.modules[i].in0)) for i in range(len(arch.modules)) if len(arch.modules[i].in0) > 1]
        mux_in1_bw = [m.math.log2_ceil(len(arch.modules[i].in1)) for i in range(len(arch.modules)) if len(arch.modules[i].in1) > 1]
        mux_in2_bw = [m.math.log2_ceil(len(arch.modules[i].in2)) for i in range(len(arch.modules)) if len(arch.modules[i].in2) > 1]


        mux_in0_asmd = [BitVector[mux_in0_bw[i]](n) for i, n in enumerate(mux_in0)]
        mux_in1_asmd = [BitVector[mux_in1_bw[i]](n) for i, n in enumerate(mux_in1)]
        mux_in2_asmd = [BitVector[mux_in2_bw[i]](n) for i, n in enumerate(mux_in2)]

        mux_out_bw = [m.math.log2_ceil(len(arch.outputs[i])) for i in range(arch.num_outputs) if len(arch.outputs[i]) > 1]
        mux_out_asmd = [BitVector[mux_out_bw[i]](n) for i, n in enumerate(mux_out)]

        
        mux_bit_out_bw = [m.math.log2_ceil(len(arch.bit_outputs[i])) for i in range(arch.num_bit_outputs) if len(arch.bit_outputs[i]) > 1]
        mux_bit_out_asmd = [BitVector[mux_bit_out_bw[i]](n) for i, n in enumerate(mux_bit_out)]


        # inst_gen = gen_inst(alu=alu, mul=mul, mux_in0=mux_in0, mux_in1=mux_in1, mux_out=mux_out)
        # input_binding.append((assembler(inst_gen), ("inst",)))

        # constraint = {}

        for ind, a in enumerate(alu):
            input_binding.append((a, ("inst", "alu", ind)))

        for ind, a in enumerate(bit_alu):
            input_binding.append((a, ("inst", "bit_alu", ind)))

        for ind, a in enumerate(cond):
            input_binding.append((a, ("inst", "cond", ind)))

        for ind, a in enumerate(mul):
            input_binding.append((a, ("inst", "mul", ind)))

        for ind, a in enumerate(mux_in0_asmd):
            input_binding.append((a, ("inst", "mux_in0", ind)))

        for ind, a in enumerate(mux_in1_asmd):
            input_binding.append((a, ("inst", "mux_in1", ind)))

        for ind, a in enumerate(mux_in2_asmd):
            input_binding.append((a, ("inst", "mux_in2", ind)))

        for ind, a in enumerate(mux_out_asmd):
            input_binding.append((a, ("inst", "mux_out", ind)))
            
        for ind, a in enumerate(mux_bit_out_asmd):
            input_binding.append((a, ("inst", "mux_bit_out", ind)))

        for ind, a in enumerate(signed):
            input_binding.append((a, ("inst", "signed", ind)))

        for ind, a in enumerate(lut):
            input_binding.append((a, ('inst', 'lut', ind)))


        constrained_vars = {"inst", "inputs"}

        for i in arch_mapper.input_varmap:
            if i[0] not in constrained_vars and "inputs" not in i[0]:
                input_binding.append((peak.mapper.utils.Unbound, i))

        new_rrule = {}

        # constraints.append(constraint)
        new_rrule["ibinding"] = input_binding

        if 'out0' in rrule:
            new_rrule["obinding"] = [((0,), ('pe_outputs_0', ))]
        else:
            new_rrule["obinding"] = [((0,), ('pe_outputs_1', ))]

        rrules_out.append(new_rrule)
 

    return rrules_out


def test_rewrite_rules(rrules):
    arch = read_arch("./outputs/PE.json")
    PE_fc = wrapped_peak_class(arch)
    arch_mapper = ArchMapper(PE_fc)
    

    for rr_ind, rrule in enumerate(rrules):

        peak_eq = importlib.import_module("outputs.peak_eqs.peak_eq_" + str(rr_ind))
        

        print("Rewrite rule ", rr_ind)
        # pretty_print_binding(rrule["ibinding"])
        # pretty_print_binding(rrule["obinding"]) 

        try:
            solution = RewriteRule(rrule["ibinding"], rrule["obinding"], peak_eq.mapping_function_fc, PE_fc)
            counter_example = solution.verify()
        except:
            print("solving for rr")
            ir_mapper = arch_mapper.process_ir_instruction(peak_eq.mapping_function_fc)
            solution = ir_mapper.solve("btor", external_loop=True)
            if solution is None:
                print("No rewrite rule found")
                exit()
            counter_example = solution.verify()

        if counter_example is not None: 
            print("solving for rr")
            ir_mapper = arch_mapper.process_ir_instruction(peak_eq.mapping_function_fc)
            solution = ir_mapper.solve("btor", external_loop=True)
            if solution is None:
                print("No rewrite rule found")
                exit()
            counter_example = solution.verify()
        else:
            print("PASSED rewrite rule verify")
        

        rrules[rr_ind] = solution
    
    return rrules

def write_rewrite_rules(rrules):
    for sub_idx, rrule in enumerate(rrules):
        serialized_rr = rrule.serialize_bindings()

        if not os.path.exists('outputs/rewrite_rules'):
            os.makedirs('outputs/rewrite_rules')

        with open("outputs/rewrite_rules/rewrite_rule_" + str(sub_idx) + ".json", "w") as write_file:
            json.dump(serialized_rr, write_file, indent=2)