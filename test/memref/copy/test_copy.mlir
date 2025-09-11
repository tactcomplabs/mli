func.func @main() -> i32 {
	%1 = memref.alloca() : memref<4xi32>
	%2 = memref.alloca() : memref<4xi32>
	%zero = arith.constant 0 : index
    %three = arith.constant 3 : i32
	%five = arith.constant 5: i32
    memref.store %three, %1[%zero] : memref<4xi32>
	memref.copy %1, %2 : memref<4xi32> to memref<4xi32>
	memref.store %five, %1[%zero] : memref<4xi32>
	%probably_three = memref.load %1[%zero] : memref<4xi32>
	%probably_five = memref.load %2[%zero] : memref<4xi32>
	%eight = arith.addi %probably_three, %probably_five : i32
	func.return %eight : i32
}
