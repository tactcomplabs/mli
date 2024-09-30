func.func @main(%arg0: i32, %arg1: i32) -> i32 {
  %add_result = llvm.add %arg0, %arg1 : i32
  %sub_result = llvm.sub %add_result, %arg1 : i32
  %or_result = llvm.or %sub_result, %arg1 : i32
  %and_result = llvm.and %or_result, %arg0 : i32
  %shl_result = llvm.shl %and_result, %arg1 : i32
  %xor_result = llvm.xor %shl_result, %arg0 : i32
  %ashr_result = llvm.ashr %shl_result, %shl_result : i32
  llvm.return %add_result : i32
}
