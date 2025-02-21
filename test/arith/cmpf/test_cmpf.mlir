// TODO: Add other predicates
func.func @main(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf ogt, %arg0, %arg1 : f32
  func.return %res : i1
}
