func.func @oeq(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf oeq, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @ogt(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf ogt, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @oge(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf oge, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @olt(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf olt, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @ole(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf ole, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @one(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf one, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @ord(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf ord, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @ueq(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf ueq, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @ugt(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf ugt, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @uge(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf uge, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @ult(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf ult, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @ule(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf ule, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @une(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf une, %arg0, %arg1 : f32
  llvm.return %res: i1
}

func.func @uno(%arg0: f32, %arg1: f32) -> i1 {
  %res = arith.cmpf uno, %arg0, %arg1 : f32
  llvm.return %res: i1
}
