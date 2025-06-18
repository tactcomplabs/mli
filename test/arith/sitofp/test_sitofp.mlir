func.func @main(%arg0 : i32) -> f32 { 
    %res = arith.sitofp %arg0 : i32 to f32
    func.return %res : f32
}
    