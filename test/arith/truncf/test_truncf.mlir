func.func @main(%arg0 : f64) -> f32 { 
    %res = arith.truncf %arg0 : f64 to f32
    func.return %res : f32
}
    