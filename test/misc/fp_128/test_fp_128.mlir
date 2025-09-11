func.func @quad_form(%a: f128, %b: f128, %c: f128) -> (f128, f128) {
  %b2 = arith.mulf %b, %b : f128
  %four = arith.constant 4.0 : f128
  %a4 = arith.mulf %four, %a : f128
  %ac4 = arith.mulf %a4, %c : f128
  %diff = arith.subf %b2, %ac4 : f128
  %disc = llvm.intr.sqrt (%diff) : (f128) -> f128

  %neg_b = arith.negf %b : f128
  %zero = arith.constant 0.0 : f128

  // Take +/- branch matching sign(b) to avoid catastrophic cancellation
  %b_is_neg = arith.cmpf "olt", %b, %zero : f128
  %numerator = scf.if %b_is_neg -> f128 {
	%diff_neg = arith.addf %neg_b, %disc : f128
	scf.yield %diff_neg : f128
  } else {
	%diff_pos = arith.subf %neg_b, %disc : f128
	scf.yield %diff_pos : f128
  }

  %two = arith.constant 2.0 : f128
  %a2 = arith.mulf %two, %a : f128
  %r1 = arith.divf %numerator, %a2 : f128

  // By Vieta, r_1 * r_2 = c/a
  %ar1 = arith.mulf %r1, %a : f128
  %r2 = arith.divf %c, %ar1 : f128
  func.return %r1, %r2 : f128, f128
}
