module {
  func.func @main() -> i32 {
    // Set up loop bounds and initial value using index types for scf.for
    %c0 = llvm.mlir.constant(0 : index) : index
    %c10 = llvm.mlir.constant(10 : index) : index
    %c1 = llvm.mlir.constant(1 : index) : index
    %init = llvm.mlir.constant(0 : i32) : i32

    // scf.for requires index types for bounds and induction variable
    %result = scf.for %i = %c0 to %c10 step %c1 
        iter_args(%sum = %init) -> (i32) {
      // First convert index to i64, then to i32
      %i_i64 = llvm.index_cast %i : index to i64
      %i_i32 = llvm.trunc %i_i64 : i64 to i32
      %next = llvm.add %sum, %i_i32 : i32
      scf.yield %next : i32
    }
    
    return %result : i32
  }
}
