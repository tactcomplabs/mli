import os
import re 

OP_NAMES = {
"addf" : [2, 'f32', 'f32'],
"addi" : [2, 'i32', 'i32'],
"addui_extended" : [2, 'i32', 'i32, i1'],
"andi" : [2, 'i32', 'i32'],
"ceildivsi" : [2, 'i32', 'i32'],
"ceildivui" : [2, 'i32', 'i32'],
"divf" : [2, 'f32', 'f32'],
"divsi" : [2, 'i32', 'i32'],
"divui" : [2, 'i32', 'i32'],
"extf" : [2, 'f32', 'f64'],
"extsi" : [2, 'i32', 'i64'],
"extui" : [2, 'i32', 'i64'],
"floordivsi" : [2, 'i32', 'i32'],
"fptosi" : [1, 'f32', 'i32'],
"fptoui" : [1, 'f32', 'i32'],
"maximumf" : [2, 'f32', 'f32'],
"maxnumf" : [2, 'f32', 'f32'],
"maxsi" : [2, 'i32', 'i32'],
"maxui" : [2, 'i32', 'i32'],
"minimumf" : [2, 'f32', 'f32'],
"minnumf" : [2, 'f32', 'f32'],
"minsi" : [2, 'i32', 'i32'],
"minui" : [2, 'i32', 'i32'],
"mulf" : [2, 'f32', 'f32'],
"muli" : [2, 'i32', 'i32'],
"mulsi_extended" : [2, 'i32', 'i32'],
"mului_extended" : [2, 'i32', 'i32'],
"negf" : [1, 'f32', 'f32'],
"ori" : [2, 'i32', 'i32'],
"remf" : [2, 'f32', 'f32'],
"remsi" : [2, 'i32', 'i32'],
"remui" : [2, 'i32', 'i32'],
"scaling_extf" : [2, 'i32', 'i32'],
"scaling_truncf" : [2, 'i32', 'i32'],
"shli" : [2, 'i32', 'i32'],
"shrsi" : [2, 'i32', 'i32'],
"shrui" : [2, 'i32', 'i32'],
"sitofp" : [1, 'i32', 'f32'],
"subf" : [2, 'f32', 'f32'],
"subi" : [2, 'i32', 'i32'],
"truncf" : [1, 'f64', 'f32'],
"trunci" : [1, 'i32', 'i8'],
"uitofp" : [1, 'i32', 'f32'],
"xori" : [2, 'i32', 'i32'],
}

def generate_test(op_name, num_params, arg_type, ret_type):
    if num_params == 1:
        signature = f"%arg0 : {arg_type}"
        func_args = "%arg0"
    elif num_params == 2:
        signature = f"%arg0: {arg_type}, %arg1 : {arg_type}"
        func_args = "%arg0, %arg1"
        
    if op_name in ['fptosi', 'fptoui', 'sitofp', 'uitofp', 'bitcast', 'truncf', 'trunci']:
        op_ret_type = f"{arg_type} to {ret_type}"
    else:
        op_ret_type = ret_type 
        
    return f"""func.func @main({signature}) -> {ret_type} {{ 
    %res = arith.{op_name} {func_args} : {op_ret_type}
    func.return %res : {ret_type}\n}}
    """

    
PREFIX = '../test/arith'
for op in OP_NAMES.keys():
    DIR_NAME = os.path.join(PREFIX, op)
    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)
        print(f"Created {DIR_NAME} folder")
    params = OP_NAMES[op]
    TEST_FILE = os.path.join(DIR_NAME, f"test_{op}.mlir")
    if True:
        with open(TEST_FILE, 'w') as fp:
            fp.write(generate_test(op, params[0], params[1], params[2]))
        print(f"Created test {TEST_FILE}")