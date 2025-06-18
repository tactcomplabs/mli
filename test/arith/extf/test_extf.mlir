func.func @main(%arg0: f32, %arg1 : f32) -> f64 { 
    %res = arith.extf %arg0, %arg1 : f64
    func.return %res : f64
}
    