func.func @eq(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi eq, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @ne(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi ne, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @slt(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi slt, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @sle(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi sle, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @sgt(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi sgt, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @sge(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi sge, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @ult(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi ult, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @ule(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi ule, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @ugt(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi ugt, %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @uge(%arg0: i32, %arg1: i32) -> i1 {
  %res = arith.cmpi uge, %arg0, %arg1 : i32
  llvm.return %res : i1
}