supported_ops =  {"mul", "const", "not", "and", "or", "xor", "shl", \
                    "lshr", "ashr", "neg", "add", "sub", \
                    "sle", "sge", "ule", "uge", "eq", "mux", \
                    "slt", "sgt", "ult", "ugt", "smax", "smin", \
                    "umax", "umin", "absd", "abs", \
                    "bitnot", "bitconst", "bitand", "bitor", "bitxor", "bitmux", \
                    "floatadd", "floatsub", "floatmul"}

bit_input_ops = {"bitand", "bitor", "bitxor", "bitnot", "bitmux"}

bit_output_ops = {"lut", "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", "bitand", "bitor", "bitxor", "bitnot", "bitmux", "bitconst"}

lut_supported_ops = {"bitand", "bitor", "bitxor", "bitnot", "bitmux"}

comm_ops =  {"and", "or", "xor", "add", "eq", "mul", "alu", "umax", "umin", "smax", "smin"}

primitive_ops = {"and", "or", "xor", "shl", "lshr", "ashr", "add", "sub",
                "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", 
                "smax", "smin", "umax", "umin", "absd", "abs", "mul", "mux",
                "bitand", "bitor", "bitxor", "bitnot", "bitmux", "floatadd", "floatsub", "floatmul"}

alu_supported_ops = {"and", "or", "xor", "shl", "lshr", "ashr", "add", "sub",
                    "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", 
                    "smax", "smin", "umax", "umin", "absd", "abs", "floatadd", "floatsub", "floatmul"}

fp_alu_supported_ops = {"floatadd", "floatsub", "floatmul"}

non_coreir_ops = {"alu", "bit_alu", "lut", "input", "bit_input", "const_input", "bit_const_input", "output", "bit_output", "gte", "lte", "sub", "shr"}

op_types = []
op_types_flipped = []

# def read_optypes():
#     with open(".temp/op_types.txt", "rb") as file:
#         op_types_from_file = pickle.load(file)

#     curr_ops = [*op_types_from_file]

#     for op in primitive_ops:
#         if op not in curr_ops:
#             curr_ops.append(op)

#     global op_types
#     op_types = {str(k): v for k, v in enumerate(curr_ops)}

#     for op in non_coreir_ops:
#         if op not in op_types:
#             op_types[op] = op

#     global op_types_flipped 
#     op_types_flipped = {v: k for k, v in op_types.items()}
