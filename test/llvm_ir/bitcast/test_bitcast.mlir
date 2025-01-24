func.func @itofp(%arg0: i32) -> f32 {
  %res = llvm.bitcast %arg0 : i32 to f32
  llvm.return %res : f32
}

func.func @fptoi(%arg0: f32) -> i32 {
  %res = llvm.bitcast %arg0 : f32 to i32
  llvm.return %res : i32
}
