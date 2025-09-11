func.func @main(%0: index, %1: index) {
	%arr = memref.alloc(%0, %1) : memref<?x?xi32>
 	memref.dealloc %arr : memref<?x?xi32>
	func.return
}
