func.func @main(%arg0: i32, %arg1: i32) -> i32 {
  %res = llvm.udiv %arg0, %arg1 : i32
  llvm.return %res : i32
}