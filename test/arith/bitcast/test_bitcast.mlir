func.func @fptoi(%arg0 : f32) -> i32 { 
    %res = arith.bitcast %arg0 : f32 to i32
    func.return %res : i32
}

func.func @itofp(%arg0 : i32) -> f32 { 
    %res = arith.bitcast %arg0 : i32 to f32
    func.return %res : f32
}
    