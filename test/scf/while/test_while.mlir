// Test while loop with multiple carried values
func.func @while(%arg0: i32) -> (i32, i32) {
  %limit = arith.constant 10 : i32
  %one = arith.constant 1 : i32
  %zero = arith.constant 0 : i32

  %result:2 = scf.while (%current = %arg0, %sum = %zero) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %current, %limit : i32
    scf.condition(%cond) %current, %sum : i32, i32
  } do {
  ^bb0(%current_val: i32, %sum_val: i32):
    %next = arith.addi %current_val, %one : i32
    %new_sum = arith.addi %sum_val, %current_val : i32
    scf.yield %next, %new_sum : i32, i32
  }
  return %result#0, %result#1 : i32, i32
}

// Test "do-while" loop with multiple carried values
func.func @do_while(%arg0: i32) -> (i32, i32) {
  %limit = arith.constant 10 : i32
  %one = arith.constant 1 : i32
  %zero = arith.constant 0 : i32

  %result:2 = scf.while (%current = %arg0, %sum = %zero) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %current, %limit : i32
	%next = arith.addi %current, %one : i32
    %new_sum = arith.addi %sum, %current : i32
    scf.condition(%cond) %next, %new_sum : i32, i32
  } do {
  ^bb0(%current_val: i32, %sum_val: i32):
        scf.yield %current_val, %sum_val : i32, i32
  }

  return %result#0, %result#1 : i32, i32
}

