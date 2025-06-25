func.func @add(%0: i32, %1: i32) -> i32 {
	%2 = llvm.add %0, %1 : i32
	func.return %2 : i32
}

func.func @main(%0: i32, %1: i32) -> i32 {
	%add_ref = func.constant @add : (i32, i32) -> i32
	%sum = func.call_indirect %add_ref(%0, %1) : (i32, i32) -> i32
	func.return %sum : i32
}
