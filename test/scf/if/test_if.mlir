// Test basic if without else
func.func @test_if_basic(%arg0: i1) -> i1 {
  scf.if %arg0 {
    // Empty then block
  }
  return %arg0 : i1
}

// Test if-else with return values
func.func @test_if_else(%arg0: i1) -> i32 {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32

  %result = scf.if %arg0 -> i32 {
    scf.yield %one : i32
  } else {
    scf.yield %zero : i32
  }
  return %result : i32
}

// Test nested if conditions
func.func @test_if_nested(%arg0: i1, %arg1: i1) -> i32 {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32

  %result = scf.if %arg0 -> i32 {
    %nested = scf.if %arg1 -> i32 {
      scf.yield %two : i32
    } else {
      scf.yield %one : i32
    }
    scf.yield %nested : i32
  } else {
    scf.yield %zero : i32
  }
  return %result : i32
}
