func.func @main(%0: index) -> index {
	%1 = memref.alloca() : memref<4x2xf32>
	%2 = memref.dim %1, %0 : memref<4x2xf32>
	func.return %2 : index
}
