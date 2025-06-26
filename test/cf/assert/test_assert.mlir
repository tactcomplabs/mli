func.func @main(%0: i1) {
	cf.assert %0, "assertion failed"
	func.return
}
