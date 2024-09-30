//===- LLVMInterpreter.h - LLVM dialect interpreter -------------*- C++ -*-===//
//
// Copyright (C) 2017-2024 Tactical Computing Laboratories, LLC
// All Rights Reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file contains the dialect interpreter declaration for the LLVM dialect.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_INTERPRETER_DIALECTS_LLVMINTERPRETER_H_
#define MLIR_INTERPRETER_DIALECTS_LLVMINTERPRETER_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {

/// Dialect interpreter for the LLVM dialect.
class LLVMInterpreter {
public:
  using Dialect = LLVM::LLVMDialect;
  using Context = void;

  /// Attach the InterpreterOpInterface to the ops in the Func dialect.
  static void attachInterface(MLIRContext &context);

  /// Create an interpreter context.
  static Context *createContext() { return nullptr; }

  /// Destroy the interpreter context.
  static void destroyContext(Context *) {}
};

} // namespace mlir

#endif // MLIR_INTERPRETER_DIALECTS_LLVMINTERPRETER_H_
