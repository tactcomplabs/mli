func.func @toindex(%arg0: i32) -> index {
    %res = arith.index_castui %arg0 : i32 to index
    func.return %res : index
}

func.func @toint(%arg0: index) -> i32 {
    %res = arith.index_castui %arg0 : index to i32
    func.return %res : i32
}
