func.func @main(%arg0: i128, %arg1: i128) -> i128 {
  %res = llvm.add %arg0, %arg1 : i128
  llvm.return %res : i128
}
