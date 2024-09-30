////===- FuncInterpreter.h - Func dialect interpreter -------------*- C++ -*-===//
//
// Part of this file is part of the LLVM Project, under the Apache License v2.0
// with LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The remaining portions of this file are:
//
// Copyright (C) 2024 Tactical Computing Laboratories, LLC
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
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERPRETER_DIALECTS_FUNCINTERPRETER_H_
#define MLIR_INTERPRETER_DIALECTS_FUNCINTERPRETER_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {

/// Dialect interpreter for the Func dialect.
class FuncInterpreter {
public:
  using Dialect = func::FuncDialect;
  using Context = void;

  /// Attach the InterpreterOpInterface to the ops in the Func dialect.
  static void attachInterface(MLIRContext &context);

  /// Create an interpreter context.
  static Context *createContext() { return nullptr; }

  /// Destroy the interpreter context.
  static void destroyContext(Context *) {}
};

} // namespace mlir

#endif // MLIR_INTERPRETER_DIALECTS_FUNCINTERPRETER_H_
