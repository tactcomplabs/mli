func.func @main(%arg0: i32, %arg1: i32) -> i32 {
  %res = "llvm.intr.smax"(%arg0, %arg1) : (i32, i32) -> i32
  llvm.return %res : i32
}