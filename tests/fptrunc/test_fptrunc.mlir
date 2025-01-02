func.func @main(%arg0: f64) -> f32 {
  %res = llvm.fptrunc %arg0 : f64 to f32
  llvm.return %res : f32
}