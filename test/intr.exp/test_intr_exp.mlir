func.func @main(%arg0: f32) -> f32 {
  %res = "llvm.intr.exp"(%arg0): (f32) -> f32
  llvm.return %res : f32
}
