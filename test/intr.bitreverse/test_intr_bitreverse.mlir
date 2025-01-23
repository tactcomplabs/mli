func.func @main(%arg0: i32) -> i32 {
  %res = "llvm.intr.bitreverse"(%arg0): (i32) -> i32
  llvm.return %res : i32
}