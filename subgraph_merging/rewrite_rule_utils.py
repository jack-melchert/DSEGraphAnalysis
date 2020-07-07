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


from peak_gen.sim import wrapped_pe_arch_closure
from peak_gen.arch import read_arch, graph_arch
from peak_gen.isa import inst_arch_closure
from peak_gen.asm import asm_arch_closure
from peak_gen.alu import ALU_t, Signed_t
from peak_gen.cond import Cond_t
from peak_gen.mul import MUL_t

def gen_rewrite_rule(node_dict):
    rr = {}
    for k, v in node_dict.items():
        rr[k] = {}
        rr[k]['0'] = v['0']
        if not (v['alu_op'] == "const" or v['alu_op'] == "output"):
            rr[k]['1'] = v['1']

        if v['alu_op'] == "mux":
            rr[k]['2'] = v['2']

        rr[k]['alu_op'] = v['alu_op']

    print(rr)
    return rr


def formulate_rewrite_rules(rrules, merged_arch):

    bit_output_ops = {"sle", "sge", "slt", "sgt", "ule", "uge", "ult", "ugt", "eq"}
    rrules_out = []

    arch = read_arch("./outputs/subgraph_archs/subgraph_arch_merged.json")
    graph_arch(arch)
    PE_fc = wrapped_pe_arch_closure(arch)
    arch_mapper = ArchMapper(PE_fc)

    for sub_idx, rrule in enumerate(rrules):
        rr_output = {}
      
        input_mappings = {}
        bit_input_mappings = {}
        const_mappings = {}
        seen_inputs = []

        alu = []
        mul = []
        ops = []
        mux_in0 = []
        mux_in1 = []
        mux_sel = []
        mux_out = []
        cfg_idx = 0

        for module in merged_arch["modules"]:
            if module["id"] in rrule:
                v = rrule[module["id"]]

                if module["type"] == "alu" or module["type"] == "mul" or module["type"] == "mux":
                    if module["type"] == "alu":
                        alu.append(v['alu_op'])
                        ops.append(v['alu_op'])
                    elif module["type"] == "mul":
                        mul.append("Mult0")
                        ops.append("Mult0")

                    if len(module["in0"]) > 1:
                        for mux_idx, mux_in in enumerate(module["in0"]):
                            if v['0'] == mux_in:
                                mux_in0.append(mux_idx)

                                if "in" in v['0']:
                                    in_idx = int(mux_in.split("in")[1])
                                    input_mappings[in_idx] = v['0']
                                    seen_inputs.append(v['0'])
                            elif "in" in mux_in:
                                in_idx = int(mux_in.split("in")[1])

                                if mux_in not in seen_inputs:
                                    input_mappings[in_idx] = 0
                                    seen_inputs.append(mux_in)

                    elif "in" in module["in0"][0]:
                        in_idx = int(v['0'].split("in")[1])
                        input_mappings[in_idx] = v['0']
                        seen_inputs.append(v['0'])

                    if len(module["in1"]) > 1:
                        for mux_idx, mux_in in enumerate(module["in1"]):
                            if v['1'] == mux_in:
                                mux_in1.append(mux_idx)

                                if "in" in v['1']:
                                    in_idx = int(mux_in.split("in")[1])
                                    input_mappings[in_idx] = v['1']
                                    seen_inputs.append(v['1'])
                            elif "in" in mux_in:
                                in_idx = int(mux_in.split("in")[1])

                                if mux_in not in seen_inputs:
                                    input_mappings[in_idx] = 0
                                    seen_inputs.append(mux_in)

                    elif "in" in module["in1"][0]:
                        in_idx = int(v['1'].split("in")[1])
                        input_mappings[in_idx] = v['1']
                        seen_inputs.append(v['1'])

                    if module["type"] == "mux":
                        if len(module["sel"]) > 1:
                            for mux_idx, mux_in in enumerate(module["sel"]):
                                if v['2'] == mux_in:
                                    mux_sel.append(mux_idx)

                                    if "in" in v['2']:
                                        in_idx = int(mux_in.split("in")[1])
                                        bit_input_mappings[in_idx] = v['2']
                                        seen_inputs.append(v['2'])
                                elif "in" in mux_in:
                                    in_idx = int(mux_in.split("in")[1])

                                    if mux_in not in seen_inputs:
                                        bit_input_mappings[in_idx] = 0
                                        seen_inputs.append(mux_in)

                        elif "in" in module["sel"][0]:
                            in_idx = int(v['2'].split("in")[1])
                            bit_input_mappings[in_idx] = v['2']
                            seen_inputs.append(v['2'])

                elif module["type"] == "const":
                    # cfg_idx = int(v['0'].split("const")[1])
                    const_mappings[cfg_idx] = v['0']
                    cfg_idx += 1


            else:
                if module["type"] == "alu" or module["type"] == "mul" or module["type"] == "mux":
                    if module["type"] == "alu":
                        alu.append("add")
                        ops.append("add")
                    elif module["type"] == "mul":
                        mul.append("Mult0") 
                        ops.append("Mult0") 
                    elif module["type"] == "mux":
                        if len(module["sel"]) > 1:
                            mux_sel.append(0)
                        for mux_in in module["sel"]:
                            if "in" in mux_in:
                                in_idx = int(mux_in.split("in")[1])

                                if mux_in not in seen_inputs:
                                    bit_input_mappings[in_idx] = 0
                                    seen_inputs.append(mux_in)

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

                elif module["type"] == "const":
                    const_mappings[cfg_idx] = 0
                    cfg_idx += 1
            

        
        output_id = rrule["out0"]["0"]
        if len(arch.outputs[0]) > 1:
            mux_out.append(arch.outputs[0].index(output_id))

        input_binding = []

        for k, v in input_mappings.items():
            if v == 0:
                input_binding.append((peak.mapper.utils.Unbound, ("inputs", k)))
            else:
                input_binding.append(((v,), ("inputs", k)))

        input_binding.append((peak.mapper.utils.Unbound, ("bit_inputs", 0)))
        input_binding.append((peak.mapper.utils.Unbound, ("bit_inputs", 1)))
        input_binding.append((peak.mapper.utils.Unbound, ("bit_inputs", 2)))

        for k, v in bit_input_mappings.items():
            if v == 0:
                input_binding.append((peak.mapper.utils.Unbound, ("bit_inputs", k+3)))
            else:
                input_binding.append(((v,), ("bit_inputs", k+3)))


        for k, v in const_mappings.items():
            if v == 0:
                input_binding.append((peak.mapper.utils.Unbound, ("inst", "const_data", k)))
            else:
                input_binding.append(((v,), ("inst", "const_data", k)))


        op_map = {}

        op_map["Mult0"] = MUL_t.Mult0
        op_map["Mult1"] = MUL_t.Mult1
        # op_map["not"] = ALU_t.Sub  
        op_map["and"] = ALU_t.And    
        op_map["or"] = ALU_t.Or    
        op_map["xor"] = ALU_t.XOr    
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
        op_map["absd"] = ALU_t.Sub

        cond_map = {}
        cond_map["Mult0"] = Cond_t.ALU
        cond_map["Mult1"] = Cond_t.ALU
        # cond_map["not"] = Cond_t.ALU  
        cond_map["and"] = Cond_t.ALU    
        cond_map["or"] = Cond_t.ALU    
        cond_map["xor"] = Cond_t.ALU    
        cond_map["shl"] = Cond_t.ALU   
        cond_map["lshr"] = Cond_t.ALU    
        cond_map["ashr"] = Cond_t.ALU    
        cond_map["neg"] = Cond_t.ALU   
        cond_map["add"] = Cond_t.ALU    
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
        cond_map["umax"] = Cond_t.ALU    
        cond_map["smax"] = Cond_t.ALU  
        cond_map["umin"] = Cond_t.ALU  
        cond_map["smin"] = Cond_t.ALU
        cond_map["abs"] = Cond_t.ALU
        cond_map["absd"] = Cond_t.ALU

        signed_map = {}
        signed_map["Mult0"] = Signed_t.unsigned
        signed_map["Mult1"] = Signed_t.unsigned
        # signed_map["not"] = Signed_t.unsigned  
        signed_map["and"] = Signed_t.unsigned    
        signed_map["or"] = Signed_t.unsigned    
        signed_map["xor"] = Signed_t.unsigned    
        signed_map["shl"] = Signed_t.unsigned   
        signed_map["lshr"] = Signed_t.unsigned    
        signed_map["ashr"] = Signed_t.unsigned    
        signed_map["neg"] = Signed_t.unsigned   
        signed_map["add"] = Signed_t.unsigned    
        signed_map["sub"] = Signed_t.unsigned    
        signed_map["sle"] = Signed_t.signed    
        signed_map["sge"] = Signed_t.signed     
        signed_map["slt"] = Signed_t.signed   
        signed_map["sgt"] = Signed_t.signed  
        signed_map["ult"] = Signed_t.unsigned    
        signed_map["ugt"] = Signed_t.unsigned    
        signed_map["ule"] = Signed_t.unsigned   
        signed_map["uge"] = Signed_t.unsigned    
        signed_map["eq"] = Signed_t.unsigned
        signed_map["umax"] = Signed_t.unsigned    
        signed_map["smax"] = Signed_t.signed  
        signed_map["umin"] = Signed_t.unsigned  
        signed_map["smin"] = Signed_t.signed
        signed_map["abs"] = Signed_t.unsigned
        signed_map["absd"] = Signed_t.unsigned

        signed = [signed_map[n] for n in ops]
        cond = [cond_map[n] for n in alu]
        alu = [op_map[n] for n in alu]
        mul = [op_map[n] for n in mul]

        mux_in0_bw = [m.math.log2_ceil(len(arch.modules[i].in0)) for i in range(len(arch.modules)) if len(arch.modules[i].in0) > 1]
        mux_in1_bw = [m.math.log2_ceil(len(arch.modules[i].in1)) for i in range(len(arch.modules)) if len(arch.modules[i].in1) > 1]
        mux_sel_bw = [m.math.log2_ceil(len(arch.modules[i].sel)) for i in range(len(arch.modules)) if len(arch.modules[i].sel) > 1]


        mux_in0_asmd = [BitVector[mux_in0_bw[i]](n) for i, n in enumerate(mux_in0)]
        mux_in1_asmd = [BitVector[mux_in1_bw[i]](n) for i, n in enumerate(mux_in1)]
        mux_sel_asmd = [BitVector[mux_sel_bw[i]](n) for i, n in enumerate(mux_sel)]

        mux_out_bw = [m.math.log2_ceil(len(arch.outputs[i])) for i in range(arch.num_outputs) if len(arch.outputs[i]) > 1]
        mux_out_asmd = [BitVector[mux_out_bw[i]](n) for i, n in enumerate(mux_out)]


        # inst_gen = gen_inst(alu=alu, mul=mul, mux_in0=mux_in0, mux_in1=mux_in1, mux_out=mux_out)
        # input_binding.append((assembler(inst_gen), ("inst",)))

        # constraint = {}

        for ind, a in enumerate(alu):
            input_binding.append((a, ("inst", "alu", ind)))

        for ind, a in enumerate(cond):
            input_binding.append((a, ("inst", "cond", ind)))

        for ind, a in enumerate(mul):
            input_binding.append((a, ("inst", "mul", ind)))

        for ind, a in enumerate(mux_in0_asmd):
            input_binding.append((a, ("inst", "mux_in0", ind)))

        for ind, a in enumerate(mux_in1_asmd):
            input_binding.append((a, ("inst", "mux_in1", ind)))

        for ind, a in enumerate(mux_sel_asmd):
            input_binding.append((a, ("inst", "mux_sel", ind)))

        for ind, a in enumerate(mux_out_asmd):
            input_binding.append((a, ("inst", "mux_out", ind)))

        for ind, a in enumerate(signed):
            input_binding.append((a, ("inst", "signed", ind)))

        input_binding.append((BitVector[8](0), ('inst', 'lut')))


        constrained_vars = {"inst", "inputs", "bit_inputs"}

        for i in arch_mapper.input_varmap:
            if i[0] not in constrained_vars:
                input_binding.append((peak.mapper.utils.Unbound, i))

        new_rrule = {}

        # constraints.append(constraint)
        new_rrule["ibinding"] = input_binding

        if rrule[rrule['out0']['0']]['alu_op'] in bit_output_ops:
            new_rrule["obinding"] = [((0,), ('pe_outputs', 1))]
        else:
            new_rrule["obinding"] = [((0,), ('pe_outputs', 0))]

        rrules_out.append(new_rrule)

    return rrules_out


def test_rewrite_rules(rrules):
    arch = read_arch("./outputs/subgraph_archs/subgraph_arch_merged.json")
    PE_fc = wrapped_pe_arch_closure(arch)
    Inst_fc = inst_arch_closure(arch)
    Inst = Inst_fc(family.PyFamily())
 

    inst_type = PE_fc(family.PyFamily()).input_t.field_dict["inst"]

    _assembler = Assembler(inst_type)
    assembler = _assembler.assemble

    asm_fc = asm_arch_closure(arch)
    gen_inst = asm_fc(family.PyFamily())


    for rr_ind, rrule in enumerate(rrules):


        arch_mapper = ArchMapper(PE_fc)

        peak_eq = importlib.import_module("outputs.peak_eqs.peak_eq_" + str(rr_ind))


        # ir_mapper = arch_mapper.process_ir_instruction(peak_eq.mapping_function_fc)

        # solution = ir_mapper.solve(external_loop=True)

        rr = RewriteRule(rrule["ibinding"], rrule["obinding"], peak_eq.mapping_function_fc, PE_fc)

        print("Rewrite rule ", rr_ind)
        # pretty_print_binding(rr.ibinding)
        # pretty_print_binding(rr.obinding)

        counter_example = rr.verify()
        
        if counter_example is not None: 
            for i in counter_example:
                for ii in i.items():
                    print(ii)
            exit()
        else:
            print("PASSED rewrite rule verify")

        rrules[rr_ind] = rr
    
    return rrules

def write_rewrite_rules(rrules):
    for sub_idx, rrule in enumerate(rrules):
        rrule_out = {}
        rrule_out["ibinding"] = []
        for t in rrule.ibinding:
            if isinstance(t[0], BitVector):
                rrule_out["ibinding"].append(tuple([{'type':'BitVector', 'width':len(t[0]), 'value':t[0].value}, t[1]]))
            elif isinstance(t[0], Bit):
                rrule_out["ibinding"].append(tuple([{'type':'Bit', 'width':1, 'value':t[0]._value}, t[1]]))
            elif t[0] == peak.mapper.utils.Unbound:
                rrule_out["ibinding"].append(tuple(["unbound", t[1]]))
            else:
                rrule_out["ibinding"].append(t)

        rrule_out["obinding"] = []
        for t in rrule.obinding:
            if t[0] == peak.mapper.utils.Unbound:
                rrule_out["obinding"].append(tuple(["unbound", t[1]]))
            else:
                rrule_out["obinding"].append(t)

        if not os.path.exists('outputs/subgraph_rewrite_rules'):
            os.makedirs('outputs/subgraph_rewrite_rules')

        with open("outputs/subgraph_rewrite_rules/subgraph_rr_" + str(sub_idx) + ".json", "w") as write_file:
            write_file.write(json.dumps(rrule_out))