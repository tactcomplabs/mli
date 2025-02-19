//===- MLIUtils.h - Multi-Level Interpreter Utilities -------------*- C++ -*-===//
//
// Copyright (C) 2017-2025 Tactical Computing Laboratories, LLC
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
#ifndef MLI_UTILS_H
#define MLI_UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"

namespace mli {

// Parse an MLIR file, detecting and handling both bytecode and text formats
mlir::OwningOpRef<mlir::ModuleOp> parseMLIRFile(llvm::StringRef filename,
                                                mlir::MLIRContext &context);

// Print function signature 
void printFunctionSignature(mlir::func::FuncOp funcOp, llvm::raw_ostream &os);

// Validate arguments against function signature
// TODO: Maybe make this handle LLVM.func or other .funcs in the future?
bool validateFunctionArguments(mlir::func::FuncOp funcOp, 
                               llvm::ArrayRef<int32_t> providedArgs,
                               std::string &errorMessage);


// ---------------------------------------------------------------------------// 
// APInt and APFloat utilities
// ---------------------------------------------------------------------------//
llvm::APInt getIntegerData(const mlir::EvalValue val, bool isSigned = false);
llvm::APFloat getFloatData(const mlir::EvalValue val);
llvm::APFloat compute(const llvm::APFloat &num, double (*func)(double));
llvm::APFloat compute(const llvm::APFloat &lhs, const llvm::APFloat &rhs, double (*func)(double, double));
llvm::APFloat compute(const llvm::APFloat &lhs, const llvm::APInt &rhs, double (*func)(double, int64_t));
llvm::APFloat::Semantics getFloatSemantics(const mlir::Type result_type);

} // namespace mli


#endif // MLI_UTILS_H

