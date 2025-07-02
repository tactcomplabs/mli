func.func @main(i64, i1) -> i64 {

^bb0(%a: i64, %cond: i1):
  cf.cond_br %cond, ^bb1(%a, %a: i64, i64), ^bb2(%a: i64)

^bb1(%3: i64, %4: i64):
  %sum = arith.addi %3, %4 : i64 
  func.return %sum : i64

^bb2(%5: i64):
  func.return %5 : i64
}
