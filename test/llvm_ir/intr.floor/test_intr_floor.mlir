func.func @main(%arg0: f32) -> f32 {
  %res = "llvm.intr.floor"(%arg0): (f32) -> f32
  llvm.return %res : f32
}