func.func @main(%cond: i1, %arg0: i32, %arg1 : i32) -> i32 { 
    %res = arith.select %cond, %arg0, %arg1 : i32
    func.return %res : i32
}
    