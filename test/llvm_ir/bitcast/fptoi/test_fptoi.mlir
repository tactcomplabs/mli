func.func @main(%arg0: f32) -> i32 {
  %res = llvm.bitcast %arg0 : f32 to i32
  llvm.return %res : i32
}
