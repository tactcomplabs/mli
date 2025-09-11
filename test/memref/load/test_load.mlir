func.func @main(%idx1: index, %idx2: index) -> i32 {
   %arr = memref.alloca() : memref<2x3xi32>
   %val = memref.load %arr[%idx1, %idx2] : memref<2x3xi32>
   func.return %val : i32
}
