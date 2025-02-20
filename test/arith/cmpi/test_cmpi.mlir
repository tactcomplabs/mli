// Add other predicates
func.func @main(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi sgt, %arg0, %arg1 : i32
  llvm.return %res : i1
}
