func.func @main(%arg0: i64) -> i32 {
  %res = llvm.trunc %arg0 : i64 to i32
  llvm.return %res : i32
}