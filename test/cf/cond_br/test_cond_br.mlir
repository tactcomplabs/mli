func.func @main(i1) -> i32 {
	^bb0(%cond: i1):
		cf.cond_br %cond, ^bb1, ^bb2
	^bb1:
		%1 = arith.constant 2: i32
		func.return %1 : i32
	^bb2:
		%2 = arith.constant 3: i32
		func.return %2: i32
}
