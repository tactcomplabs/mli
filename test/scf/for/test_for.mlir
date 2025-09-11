// Test basic for loop with index type
func.func @test_for_basic() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %c10 step %c1 {
    // Empty body
  }
  return
}

// Test for loop with carried value
func.func @test_for_with_carried_value() -> i32 {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %one = arith.constant 1 : i32

  // Sum from 0 to 9
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%sum = %init) -> i32 {
    %next_sum = arith.addi %sum, %one : i32
    scf.yield %next_sum : i32
  }
  return %result : i32
}
