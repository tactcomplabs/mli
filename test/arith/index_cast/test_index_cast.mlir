func.func @main(%arg0: i32) -> i32 {\n\t %res = arith.index_cast %arg0 : i32\n\t func.return %res : i32\n }
