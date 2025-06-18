func.func @main(%arg0 : f32) -> i32 { 
    %res = arith.fptosi %arg0 : f32 to i32
    func.return %res : i32
}
    