func.func @main(%arg0: i32) -> i64 { 
    %res = arith.extsi %arg0 : i32 to i64
    func.return %res : i64
}
    