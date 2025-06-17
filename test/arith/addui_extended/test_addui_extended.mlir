func.func @main(%arg0: i32, %arg1: i32) -> (i32, i1) {
  %res, %overflow = arith.addui_extended %arg0, %arg1 : i32, i1
  func.return %res, %overflow : i32, i1
}
