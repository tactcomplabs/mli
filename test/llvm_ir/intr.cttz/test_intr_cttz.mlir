func.func @main(%arg0: i32) -> i32 {
  %res = "llvm.intr.cttz"(%arg0) <{is_zero_poison=0 : i1}>: (i32) -> i32
  llvm.return %res : i32
}