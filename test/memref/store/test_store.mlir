func.func @main(%val: i32, %idx1: index, %idx2: index) {
   %arr = memref.alloca() : memref<2x2xi32>
   memref.store %val, %arr[%idx1, %idx2] : memref<2x2xi32>
   memref.store %val, %arr[%idx2, %idx1] : memref<2x2xi32>
   func.return
}
