func.func @int() -> !llvm.void {
%0 = arith.constant 42 : i32
  llvm.return %0 : i32
}

func.func @float() -> !llvm.void {
  %0 = arith.constant 2.5: f32
  llvm.return %0 : f32
}
