func.func @main(%arg0: i32) -> i8 {
  %res = llvm.trunc %arg0 : i32 to i8
  llvm.return %res : i8
}
