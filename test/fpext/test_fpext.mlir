func.func @main(%arg0: f32) -> f64 {
  %res = llvm.fpext %arg0 : f32 to f64
  llvm.return %res : f64
}