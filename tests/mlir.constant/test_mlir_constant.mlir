func.func @int() -> !llvm.void {
  %0 = llvm.mlir.constant(42: i32) : i32
  llvm.return
}

func.func @float() -> !llvm.void {
  %0 = llvm.mlir.constant(2.5: f32) : f32
  llvm.return
}
