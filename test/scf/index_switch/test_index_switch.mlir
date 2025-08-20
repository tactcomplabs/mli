func.func @test_switch_with_yield(%arg0: index) -> i32 {
%0 = scf.index_switch %arg0 -> i32
case 2 {
  %1 = arith.index_cast %arg0 : index to i32
  scf.yield %1 : i32
}
default {
  %2 = arith.constant 20 : i32
  scf.yield %2 : i32
}
func.return %0 : i32
}

func.func @test_switch_without_yield(%arg0: index) {
scf.index_switch %arg0
case 2 {
  %1 = arith.index_cast %arg0 : index to i32
  scf.yield
}
default {
  %2 = arith.constant 20 : i32
  scf.yield
}
func.return
}

