func.func @main(%arg0 : i32) -> i8 { 
    %res = arith.trunci %arg0 : i32 to i8
    func.return %res : i8
}
    