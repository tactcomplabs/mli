func.func @main(i64, i1) -> i64 {

// Traditionally, a branch instruction must supply block arguments
// However, if block A dominates block B, block B can use all SSA vals declared in A without explicitly passing them
^bb0(%a: i64, %cond: i1):
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  cf.br ^bb2

^bb2: // dominated by ^bb0
  %double = arith.addi %a, %a : i64
  cf.br ^bb3

^bb3: // dominated by ^bb2
  func.return %double : i64
}
