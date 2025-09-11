func.func @main(%new_dim_1 : index, %new_dim_2 : index) {
   %arr = memref.alloca() : memref<3x3x2xf32>
   %new_shape = memref.alloca() : memref<2xindex>
   %zero = arith.constant 0 : index
   %one = arith.constant 1 : index
   memref.store %new_dim_1, %new_shape[%zero] : memref<2xindex>
   memref.store %new_dim_2, %new_shape[%one] : memref<2xindex>
   %new_arr = memref.reshape %arr(%new_shape) : (memref<3x3x2xf32>, memref<2xindex>) -> memref<?x?xf32>
   func.return
}

func.func @realloc() {
	%zero = arith.constant 0 : index
	%one = arith.constant 1 : index
	%two = arith.constant 2 : index
	%three = arith.constant 3 : index
	%arr = memref.alloc() : memref<6x3xi32>
	%shape = memref.alloc() : memref<3xindex>
	memref.store %two, %shape[%zero] : memref<3xindex>
	memref.store %three, %shape[%one] : memref<3xindex>
	memref.store %three, %shape[%two] : memref<3xindex>
	%new_arr = memref.reshape %arr(%shape) : (memref<6x3xi32>, memref<3xindex>) -> memref<*xi32> // should call realloc
	func.return
}

