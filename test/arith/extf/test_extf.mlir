func.func @main(%arg0: f16) -> f32 { 
    %res = arith.extf %arg0 : f16 to f32
    func.return %res : f32
}
    