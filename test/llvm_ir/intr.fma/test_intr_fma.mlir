func.func @main(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
  %res = "llvm.intr.fma"(%arg0, %arg1, %arg2) : (f32, f32, f32) -> f32
  llvm.return %res : f32
}