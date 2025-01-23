func.func @main(%arg0: i32) -> i32 {
  %res = "llvm.intr.abs"(%arg0) <{is_int_min_poison=0 : i1}> : (i32) -> i32
  llvm.return %res : i32
}
