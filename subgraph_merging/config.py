supported_ops =  {"mul", "mult_middle", "const", "not", "and", "or", "xor", "shl", \
                    "lshr", "ashr", "neg", "add", "sub", \
                    "sle", "sge", "ule", "uge", "eq", "mux", \
                    "slt", "sgt", "ult", "ugt", "smax", "smin", \
                    "umax", "umin", "absd", "abs", \
                    "bitnot", "bitconst", "bitand", "bitor", "bitxor", "bitmux", \
                    "floatadd", "floatsub", "floatmul"}

bit_input_ops = {"bitand", "bitor", "bitxor", "bitnot", "bitmux"}

bit_output_ops = {"lut", "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", "bitand", "bitor", "bitxor", "bitnot", "bitmux", "bitconst"}

lut_supported_ops = {"bitand", "bitor", "bitxor", "bitnot", "bitmux"}

comm_ops =  {"and", "or", "xor", "add", "eq", "mul", "mult_middle","alu", "umax", "umin", "smax", "smin"}

primitive_ops = {"and", "or", "xor", "shl", "lshr", "ashr", "add", "sub",
                "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", 
                "smax", "smin", "umax", "umin", "absd", "abs", "mul", "mux",
                "bitand", "bitor", "bitxor", "bitnot", "bitmux", "floatadd", "floatsub", "floatmul"}

alu_supported_ops = {"and", "mult_middle", "or", "xor", "shl", "lshr", "ashr", "add", "sub",
                    "sle", "sge", "ule", "uge", "eq", "slt", "sgt", "ult", "ugt", 
                    "smax", "smin", "umax", "umin", "absd", "abs", "floatadd", "floatsub", "floatmul"}

fp_alu_supported_ops = {"floatadd", "floatsub", "floatmul"}

input_names = {"input", "bit_input"}
const_names = {"const_input", "bit_const_input"}
output_names = {"output", "bit_output"}


non_coreir_ops = {"alu", "bit_alu", "lut", "input", "bit_input", "const_input", "bit_const_input", "output", "bit_output", "gte", "lte", "sub", "shr", "mult_middle",}

weights = {"const":1, "bitconst":1, "and":1, "or":1, "xor":1, "shl":1, "lshr":1, "ashr":1, "add":1, "sub":1,
    "sle":1, "sge":1, "ule":1, "uge":1, "eq":1, "slt":1, "sgt":1, "ult":1, "ugt":1, 
    "smax":2, "smin":2, "umax":2, "umin":2, "absd":4, "abs":3, "mul":1, "mult_middle":1000, "mux":1,
    "bitand":1, "bitor":1, "bitxor":1, "bitnot":1, "bitmux":1, "floatadd":1, "floatsub":1, "floatmul":1, "bit_alu":1,
    "gte":1, "lte":1, "sub":1, "shr":1}



op_costs = {
"add":	{"crit_path": 0.33, "area": 117.04, "energy": 96.62},
"sub":	{"crit_path": 0.33, "area": 117.04, "energy": 96.62},
"bit_alu":	{"crit_path": 0.22, "area": 123.16, "energy": 64.11},
"gte":	{"crit_path": 0.34, "area": 146.57, "energy": 88.66},
"shl":	{"crit_path": 0.34, "area": 104.01, "energy": 44.82},
"shr":	{"crit_path": 0.39, "area": 242.86, "energy": 107.25},
"mul":	{"crit_path": 1.00, "area": 2446.14, "energy": 2540.00},
"mult_middle":	{"crit_path": 1.00, "area": 2446.14, "energy": 2540.00},
"lte":	{"crit_path": 0.45, "area": 155.61, "energy": 88.66},
"alu":	{"crit_path": 1.00, "area": 1581.64, "energy": 1016.04},
"abs":	{"crit_path": 0.04, "area": 12.77, "energy": 10.12},
"absd":	{"crit_path": 0.65, "area": 264.67, "energy": 243.81},
"float_alu":	{"crit_path": 1.00, "area": 2446.14, "energy": 2540.00},
"const":	{"crit_path": 0.03, "area": 5, "energy": 5},
"bitconst":	{"crit_path": 0.03, "area": 5, "energy": 5},
"lut":	{"crit_path": 0.34, "area": 104.01, "energy": 44.82},
"mux":	{"crit_path": 0.34, "area": 30, "energy": 20}
}

op_map = {"const": "const",
"bitconst": "bitconst",
"and": "bit_alu",
"or": "bit_alu",
"xor": "bit_alu",
"shl": "shl",
"lshr": "shr",
"ashr": "shr",
"add": "add",
"sub": "sub",
"sle": "lte",
"sge": "gte",
"ule": "lte",
"uge": "gte",
"eq": "sub",
"slt": "sub",
"sgt": "sub",
"ult": "sub",
"ugt": "sub",
"smax": "gte",
"smin": "lte",
"umax": "gte",
"umin": "lte",
"absd": "absd",
"abs": "abs",
"mul": "mul",
"mult_middle": "mul",
"alu": "alu",
"mux": "mux",
"bitand": "lut",
"bitor": "lut",
"bitxor": "lut",
"bitnot": "lut",
"bitmux": "lut",
"floatadd": "float_alu",
"floatsub": "float_alu",
"floatmul": "float_alu",
"bit_alu": "bit_alu",
"gte": "gte",
"lte": "lte",
"sub": "sub",
"shr": "shr",
"lut": "lut"}


# op_inputs = {"const":0, "bitconst":0, "and":2, "or":2, "xor":2, "shl":2, "lshr":2, "ashr":2, "add":2, "sub":2,
#     "sle":2, "sge":2, "ule":2, "uge":2, "eq":2, "slt":2, "sgt":2, "ult":2, "ugt":2, 
#     "smax":2, "smin":2, "umax":2, "umin":2, "absd":2, "abs":1, "mul":2, "mux":3,
#     "bitand":3, "bitor":3, "bitxor":3, "bitnot":3, "bitmux":3, "floatadd":2, "floatsub":2, "floatmul":2, "bit_alu":2,
#     "gte":2, "lte":2, "sub":2, "shr":2}

op_bitwidth = {"const": [], "bitconst": [], "and": [16, 16], "or": [16, 16], "xor": [16, 16], "shl": [16, 16], "lshr": [16, 16], "ashr": [16, 16], "add": [16, 16], "sub": [16, 16],
    "sle": [16, 16], "sge": [16, 16], "ule": [16, 16], "uge": [16, 16], "eq": [16, 16], "slt": [16, 16], "sgt": [16, 16], "ult": [16, 16], "ugt": [16, 16], 
    "smax": [16, 16], "smin": [16, 16], "umax": [16, 16], "umin": [16, 16], "absd": [16, 16], "abs": [16, 16], "mul": [16, 16], "mult_middle": [16, 16], "mux": [16, 16, 1],
    "bitand": [1, 1, 1], "bitor": [1, 1, 1], "bitxor": [1, 1, 1], "bitnot": [1, 1, 1], "bitmux": [1, 1, 1], "floatadd": [16, 16], "floatsub": [16, 16], "floatmul": [16, 16], "bit_alu": [16, 16],
    "gte": [16, 16], "lte": [16, 16], "sub": [16, 16], "shr": [16, 16]}

op_types = []
op_types_flipped = []

node_counter = 0


op_str_map = {}

op_str_map["mul"] = "Data(UInt(in_0) * UInt(in_1))"
op_str_map["mult_middle"] = "(Data32(in_0) * Data32(in_1))[8:24]"
# op_str_map["mult_middle"] = "Data(UInt(in_0) ^ UInt(in_1))"
op_str_map["not"] = "Data(~UInt(in_0))"
op_str_map["and"] = "Data(UInt(in_0) & UInt(in_1))"
op_str_map["or"] = "Data(UInt(in_0) | UInt(in_1))"
op_str_map["xor"] = "Data(UInt(in_0) ^ UInt(in_1))"
op_str_map["shl"] = "Data(UInt(in_0) << UInt(in_1))"
op_str_map["lshr"] = "Data(UInt(in_0) >> UInt(in_1))"
op_str_map["ashr"] = "Data(SInt(in_0) >> SInt(in_1))"
op_str_map["neg"] = "Data(-UInt(in_0))"
op_str_map["add"] = "Data(UInt(in_0) + UInt(in_1))"
op_str_map["sub"] = "Data(UInt(in_0) - UInt(in_1))"
op_str_map["sle"] = "Bit(SInt(in_0) <= SInt(in_1))"
op_str_map["sge"] = "Bit(SInt(in_0) >= SInt(in_1))"
op_str_map["ule"] = "Bit(UInt(in_0) <= UInt(in_1))"
op_str_map["uge"] = "Bit(UInt(in_0) >= UInt(in_1))"
op_str_map["eq"] = "Bit(UInt(in_0) == UInt(in_1))"
op_str_map["slt"] = "Bit(SInt(in_0) < SInt(in_1))"
op_str_map["sgt"] = "Bit(SInt(in_0) > SInt(in_1))"
op_str_map["ult"] = "Bit(UInt(in_0) < UInt(in_1))"
op_str_map["ugt"] = "Bit(UInt(in_0) > UInt(in_1))"
op_str_map["mux"] = "Data(in_2.ite(UInt(in_1),UInt(in_0)))"
op_str_map["umax"] = "Data((UInt(in_0) >= UInt(in_1)).ite(UInt(in_0), UInt(in_1)))"
op_str_map["umin"] = "Data((UInt(in_0) <= UInt(in_1)).ite(UInt(in_0), UInt(in_1)))"
op_str_map["smax"] = "Data((SInt(in_0) >= SInt(in_1)).ite(SInt(in_0), SInt(in_1)))"
op_str_map["smin"] = "Data((SInt(in_0) <= SInt(in_1)).ite(SInt(in_0), SInt(in_1)))"
op_str_map["abs"] = "Data((SInt(in_0) >= SInt(0)).ite(SInt(in_0), SInt(in_0)*SInt(-1)))"
op_str_map["bitand"] = "Bit(Bit(in_0) & Bit(in_1))"
op_str_map["bitnot"] = "Bit(~in0)" 
op_str_map["bitor"] = "Bit(Bit(in_0) | Bit(in_1))"
op_str_map["bitxor"] = "Bit(Bit(in_0) ^ Bit(in_1))"
op_str_map["bitmux"] = "Bit(in_2.ite(Bit(in_1), Bit(in_0)))"
op_str_map["floatmul"] = "Data(UInt(in_0) * UInt(in_1))"
op_str_map["floatadd"] = "Data(UInt(in_0) + UInt(in_1))"
op_str_map["floatsub"] = "Data(UInt(in_0) - UInt(in_1))"