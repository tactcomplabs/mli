func.func @main(%arg0: i32) -> i64 {
  %res = llvm.zext %arg0 : i32 to i64
  llvm.return %res : i64
}