func.func @main(%arg0: i32) -> f32 {
  %res = llvm.bitcast %arg0 : i32 to f32
  llvm.return %res : f32
}

