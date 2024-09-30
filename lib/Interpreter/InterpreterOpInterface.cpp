#include "mlir/Interpreter/InterpreterOpInterface.h"

#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

/// Include the definitions of the interpreter op interface.
#include "mlir/Interpreter/InterpreterOpInterface.cpp.inc"

using namespace mlir;

EvalError::EvalError(llvm::StringRef message, bool shouldPrintStackTrace)
    : message(message.str()) {
  if (shouldPrintStackTrace) {
    llvm::raw_string_ostream stream(stacktrace);
    llvm::sys::PrintStackTrace(stream);
  }
}
