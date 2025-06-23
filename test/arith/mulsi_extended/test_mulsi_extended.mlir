func.func @main(%arg0: i4, %arg1 : i4) -> (i4, i4) { 
    %low, %high = arith.mulsi_extended %arg0, %arg1 : i4
    func.return %low, %high : i4, i4
}
    
