func.func @main(%arg0: f32, %arg1: f32) -> f32 {
  %res = "llvm.intr.maximum"(%arg0, %arg1) : (f32, f32) -> f32
  llvm.return %res : f32
}