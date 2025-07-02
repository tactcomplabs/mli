func.func @main(%flag: i32, %a: i32) -> i32 {
  cf.switch %flag : i32, [
	default: ^bb1(%a: i32),
    2: ^bb2(%a, %a: i32, i32),
    3: ^bb3(%a, %a, %a: i32, i32, i32)
  ]

  ^bb1(%a_1: i32):
	func.return %a: i32

  ^bb2(%a_2: i32, %b_2: i32):
	%sum = arith.addi %a_2, %b_2 : i32
	func.return %sum : i32

  ^bb3(%a_3: i32, %b_3: i32, %c_3: i32):
	%sum_1 = arith.addi %a_3, %b_3 : i32
	%sum_2 = arith.addi %sum_1, %c_3 : i32
	func.return %sum_2 : i32
}
