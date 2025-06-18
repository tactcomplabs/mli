func.func @main(%arg0: i32, %arg1 : i32) -> i64 { 
    %res = arith.extui %arg0, %arg1 : i64
    func.return %res : i64
}
    