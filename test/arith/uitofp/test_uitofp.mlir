func.func @main(%arg0 : i32) -> f32 { 
    %res = arith.uitofp %arg0 : i32 to f32
    func.return %res : f32
}
    