func.func @main(%arg0: f32, %arg1: f32) -> f32 {
  %res = arith.maximumf %arg0, %arg1 : f32
  func.return %res : f32
}
