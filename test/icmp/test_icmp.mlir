func.func @eq(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "eq" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @ne(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "ne" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @slt(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "slt" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @sle(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "sle" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @sgt(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "sgt" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @sge(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "sge" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @ult(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "ult" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @ule(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "ule" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @ugt(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "ugt" %arg0, %arg1 : i32
  llvm.return %res : i1
}

func.func @uge(%arg0: i32, %arg1: i32) -> i1 {
  %res = llvm.icmp "uge" %arg0, %arg1 : i32
  llvm.return %res : i1
}

