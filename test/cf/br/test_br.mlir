func.func @main() -> i32 {
	^bb0:
		%0 = arith.constant 1 : i32
		%3 = arith.constant 3: i32
		cf.br ^bb1(%0, %3 : i32,i32)
	^bb1(%1: i32, %4: i32):
		%2 = arith.addi %1, %1 : i32
		func.return %2 : i32
}
