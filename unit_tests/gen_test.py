import os 
import json 

def parse_all_ops(file):
    with open(file, 'r') as fp:
        ops = json.load(fp)
    return ops 

def generate_mlir_file(op_name, data, file):
    lhs_type, num_ops = data['LHS_TYPE'], data['OPS']
    ret_type = lhs_type 
    if num_ops > 1:
        rhs_type = data['RHS_TYPE']
        if 'RET_TYPE' in data.keys():
            ret_type = data['RET_TYPE']
        
    if num_ops == 1:
        text = (
            f'func.func @main(%arg0: {lhs_type}) -> {ret_type} {{\n'
            f'  %res = llvm.{op_name} %arg0: {ret_type}\n'
            f'  llvm.return %res : {ret_type}\n'
            '}'
        )
    elif num_ops == 2:
        text = (
            f'func.func @main(%arg0: {lhs_type}, %arg1: {rhs_type}) -> {ret_type} {{\n'
            f'  %res = llvm.{op_name} %arg0, %arg1 : {ret_type}\n'
            f'  llvm.return %res : {ret_type}\n'
            '}'
        )
    elif num_ops == 3: # FMA
         text = (
            f'func.func @main(%arg0: {lhs_type}, %arg1: {rhs_type}, %arg2: {ret_type}) -> {ret_type} {{\n'
            f'  %res = llvm.{op_name} %arg0, %arg1, %arg2 : {ret_type}\n'
            f'  llvm.return %res : {ret_type}\n'
            '}'
        )       
        
    with open(file, 'w') as fp:
        fp.write(text)
      
def compute_result(op_name, lhs, rhs=None, ret=None):
    if op_name == 'FNEG':
        return -lhs 
    elif op_name == 'ADD' or op_name == 'FADD':
        return lhs + rhs 
    elif op_name == 'SUB' or op_name == 'FSUB':
        return lhs - rhs 
    elif op_name == 'MUL' or op_name == 'FMUL':
        return lhs * rhs             
    elif op_name == 'SDIV' or op_name == 'UDIV':
        return lhs // rhs
    elif op_name == 'FDIV':
        return lhs / rhs 
    elif op_name == 'SHL':
        return lhs << rhs 
    elif op_name == 'LSHR':
        return (lhs >> rhs) & (1 << rhs - 1)
    elif op_name == 'ASHR':
        return lhs >> rhs 
    elif op_name == 'AND':
        return lhs & rhs 
    elif op_name == 'OR':
        return lhs | rhs 
    elif op_name == 'XOR':
        return lhs ^ rhs 
    elif op_name == 'TRUNC':
        return lhs & (1 << 32 - 1)
    elif op_name == 'ZEXT':
        return lhs # basically a no-op
    elif op_name == 'SEXT':
        sign_bit = 1 << (64 - 1)
        return (lhs & (sign_bit - 1)) - (lhs & sign_bit)
    elif op_name == 'FPTOUI' or op_name == 'FPTOSI':
        return int(lhs)
    elif op_name == 'UITOFP' or op_name == 'SITOFP':
        return float(lhs)
    elif op_name == 'FPEXT' or op_name == 'FPTRUNC' or op_name == 'BITCAST':
        return lhs # FIXME: Dunno 
    elif op_name == 'ICMP' or op_name == 'FCMP':
        # TODO: This actually takes in an enum of options for comparison
        # Default is equality 
        return lhs == rhs
    elif op_name == 'INTRIN.ABS':
        return abs(lhs)
    elif op_name == 'INTRIN.SMAX' or op_name == 'INTRIN.UMAX':
        return max(lhs, rhs)
    elif op_name == 'INTRIN.SMIN' or op_name == 'INTRIN.UMIN':
        return min(lhs, rhs)
def generate_cmake_file(op_name, data, file):
    lhs_type, num_ops = data['LHS_TYPE'], data['OPS']
    ret_type = lhs_type 
    rhs_type = None
    if num_ops > 1:
        rhs_type = data['RHS_TYPE']
        if 'RET_TYPE' in data.keys():
            ret_type = data['RET_TYPE']
    lhs, rhs, ret = 1,2,3
    if lhs_type in ['f32', 'f64']:
        lhs = 1.5
    if rhs_type in ['f32', 'f64']:
        rhs = 2.25
    if ret_type in ['f32', 'f64']:
        ret = 3.125
    
    args = f'{lhs}'
    if num_ops == 2:
        args = f'{lhs},{rhs}'
    elif num_ops == 3:
        args = f'{lhs},{rhs},{ret}'
        
    res = compute_result(op_name, lhs, rhs, ret)
    text = (
        f'if(ENABLE_TESTING)\n'
        f'   add_test(NAME {op_name.upper()}\n'
        '       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}\n'
        f'      COMMAND mli test_{op_name.replace('.', '_')}.mlir --args={args}\n'
        f'   )\n'
        f'   set_tests_properties({op_name.upper()}\n'
        f'      PROPERTIES\n'
        f'      TIMEOUT 60\n'
        f'      LABELS "all"\n'
        f'      PASS_REGULAR_EXPRESSION "result: {res}"\n'
        f'   )\n'
        f'endif()'
    )
    
    with open(file, 'w') as fp:
        fp.write(text)
        
    
    

def generate_top_level_cmake_file(ops):
    text = 'if(ENABLE_TESTING)\n'
    for op in ops:
        text += f'   add_subdirectory({op})\n'
    text += 'endif()'
    
    with open('CMakeLists.txt', 'w') as fp:
        fp.write(text)

def main():
    OPS_PATH = './all_ops.json'
    ops = parse_all_ops(OPS_PATH)
    generate_top_level_cmake_file(ops)
    for op in ops:
        data = ops[op]
        op = op.lower()
        dir_prefix = f'./{op}'
        if not os.path.exists(dir_prefix):
            os.mkdir(dir_prefix)
        generate_mlir_file(op, data, f'{dir_prefix}/test_{op.replace('.', '_')}.mlir')
        generate_cmake_file(op, data, f'{dir_prefix}/CMakeLists.txt')
        
main()
