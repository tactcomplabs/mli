module {
  func.func @main() -> i32 {
    // Set up loop bounds and initial value using index types for scf.for
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0 : i32

    // scf.for requires index types for bounds and induction variable
    %result = scf.for %i = %c0 to %c10 step %c1
        iter_args(%sum = %init) -> (i32) {
      %i_32 = arith.index_cast %i : index to i32
      %next = llvm.add %sum, %i_32 : i32
      scf.yield %next : i32
    }

    return %result : i32
  }
}
