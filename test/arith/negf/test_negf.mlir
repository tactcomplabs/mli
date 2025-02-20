func.func @main(%arg0: f32) -> f32 {
  %res = arith.negf %arg0 : f32
  func.return %res : f32
}
