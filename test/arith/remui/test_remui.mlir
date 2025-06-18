func.func @main(%arg0: i32, %arg1 : i32) -> i32 { 
    %res = arith.remui %arg0, %arg1 : i32
    func.return %res : i32
}
    