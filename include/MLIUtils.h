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

#include "MLIFormat.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using Semantics = APFloatBase::Semantics;

namespace mli {

/// Parse an MLIR file, detecting and handling both bytecode and text formats.
inline mlir::OwningOpRef<mlir::ModuleOp> parseMLIRFile(llvm::StringRef filename,
                                                       mlir::MLIRContext &context) {
  // Open the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << mli::fmt::error(errorMessage) << "\n";
    return nullptr;
  }

  // Create a source manager for the input file.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // Create parser config.
  mlir::ParserConfig config(&context);

  // Attempt to parse the file (text or bytecode).
  if (auto owning_module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, config)) {
    llvm::outs() << mli::fmt::success("Parsed MLIR file: ") << filename << "\n";
    return owning_module;
  }

  llvm::errs() << mli::fmt::error("Failed to parse MLIR file") << filename << "\n";
  return nullptr;
}

/// Print a function's signature to the provided output stream.
inline void printFunctionSignature(mlir::func::FuncOp funcOp, llvm::raw_ostream &os) {
  os << mli::fmt::dim("Function '") << mli::fmt::highlight(funcOp.getName().str())
     << mli::fmt::dim("' signature:") << "\n";
  os << "  " << mli::fmt::highlight(funcOp.getName().str()) << "(";
  
  mlir::FunctionType fnType = funcOp.getFunctionType();
  for (size_t i = 0; i < fnType.getNumInputs(); ++i) {
    if (i > 0)
      os << ", ";
    std::string typeStr;
    llvm::raw_string_ostream typeOs(typeStr);
    fnType.getInput(i).print(typeOs);
    os << mli::fmt::type(typeStr);
  }
  os << ") -> ";
  
  if (fnType.getNumResults() == 0) {
    os << mli::fmt::type("void");
  } else {
    for (size_t i = 0; i < fnType.getNumResults(); ++i) {
      if (i > 0)
        os << ", ";
      std::string typeStr;
      llvm::raw_string_ostream typeOs(typeStr);
      fnType.getResult(i).print(typeOs);
      os << mli::fmt::type(typeStr);
    }
  }
  os << "\n";
}

/// Validate function arguments against its signature.
/// Returns true if the provided arguments match the expected function signature.
/// In case of a mismatch, `errorMessage` is set accordingly.
inline bool validateFunctionArguments(mlir::func::FuncOp funcOp, 
                                        llvm::ArrayRef<int32_t> providedArgs,
                                        std::string &errorMessage) {
  // Get function type.
  mlir::FunctionType fnType = funcOp.getFunctionType();
  
  std::string signature;
  llvm::raw_string_ostream os(signature);
  printFunctionSignature(funcOp, os);
  
  // Check number of arguments.
  if (fnType.getNumInputs() != providedArgs.size()) {
    errorMessage = mli::fmt::error("Argument count mismatch") + "\n" +
                   signature +
                   mli::fmt::dim("Provided ") + mli::fmt::highlight(std::to_string(providedArgs.size())) +
                   mli::fmt::dim(" argument(s): [");

    // Print provided arguments with their inferred type.
    for (size_t i = 0; i < providedArgs.size(); ++i) {
      if (i > 0)
          errorMessage += ", ";
      errorMessage += mli::fmt::highlight(std::to_string(providedArgs[i]));
    }
    errorMessage += mli::fmt::dim("] -> ") + mli::fmt::type("i32") + "\n" +
                    mli::fmt::dim("Expected ") + mli::fmt::highlight(std::to_string(fnType.getNumInputs())) +
                    mli::fmt::dim(" argument(s)");
    return false;
  }

  // Check argument types.
  for (size_t i = 0; i < fnType.getNumInputs(); ++i) {
    mlir::Type argType = fnType.getInput(i);
    // TODO: implement floating point types.
    if (!mlir::isa<mlir::IntegerType>(argType)) {
      errorMessage = mli::fmt::error("Type mismatch for argument " + std::to_string(i)) + "\n" +
                     signature +
                     mli::fmt::dim("Provided value: ") + mli::fmt::highlight(std::to_string(providedArgs[i])) +
                     mli::fmt::dim(" -> ") + mli::fmt::type("i32") + "\n" +
                     mli::fmt::dim("Expected type: ") +
                     mli::fmt::type(std::to_string(mlir::cast<mlir::IntegerType>(argType).getWidth()) + "-bit integer");
      return false;
    }

    // Check integer width matches our int32_t.
    auto intType = mlir::cast<mlir::IntegerType>(argType);
    if (intType.getWidth() != 32) {
      errorMessage = mli::fmt::error("Integer width mismatch for argument " + std::to_string(i)) + "\n" +
                     signature +
                     mli::fmt::dim("Provided value: ") + mli::fmt::highlight(std::to_string(providedArgs[i])) +
                     mli::fmt::dim(" -> ") + mli::fmt::type("i32") + "\n" +
                     mli::fmt::dim("Expected: ") +
                     mli::fmt::type(std::to_string(intType.getWidth()) + "-bit integer");
      return false;
    }
  }
  return true;
}

/// Get integer data from an mlir::EvalValue.
inline APInt getIntegerData(const mlir::EvalValue& val, bool isSigned = false) {
  return val.getData<APInt>().front();
}

/// Get float data from an mlir::EvalValue.
inline APFloat getFloatData(const mlir::EvalValue& val) {
  return val.getData<APFloat>().front();
}

/// Get semantics for a given MLIR type.
/// These semantics are used in constructing/converting APFloat types.
inline Semantics getFloatSemantics(const mlir::Type result_type) {
  if (result_type.isF16()) { // 16-bit float.
    return Semantics::S_IEEEhalf;
  } else if (result_type.isBF16()) { // 16-bit brain float.
    return Semantics::S_BFloat;
  } else if (result_type.isF32()) { // Standard 32-bit float.
    return Semantics::S_IEEEsingle;
  }
  return Semantics::S_IEEEdouble;
}

/// Compute a unary floating point operation (e.g. std::sqrt, std::cos).
/// The result will maintain the original number's semantics.
inline APFloat compute(const APFloat &num, double (*func)(double)) {
  double tmp = num.convertToDouble();
  tmp = func(tmp);
  APFloat result = APFloat(tmp);
  bool losesInfo;
  result.convert(num.getSemantics(), llvm::RoundingMode::TowardZero, &losesInfo);
  return result;
}

/// Compute a binary floating point operation (e.g. std::pow, std::fmax)
/// on two floats, converting them to double for computation.
inline APFloat compute(const APFloat &lhs, const APFloat &rhs,
                              double (*func)(double, double)) {
  double tmp1 = lhs.convertToDouble();
  double tmp2 = rhs.convertToDouble();
  tmp1 = func(tmp1, tmp2);
  APFloat result = APFloat(tmp1);
  bool losesInfo;
  result.convert(lhs.getSemantics(), llvm::RoundingMode::TowardZero, &losesInfo);
  return result;
}

/// Compute a binary operation (e.g. std::powi) between a float and an integer.
/// The float is converted to double for computation.
inline APFloat compute(const APFloat &lhs, const APInt &rhs,
                              double (*func)(double, int64_t)) {
  double tmp1 = lhs.convertToDouble();
  int64_t tmp2 = rhs.getSExtValue();
  tmp1 = func(tmp1, tmp2);
  APFloat result = APFloat(tmp1);
  bool losesInfo;
  result.convert(lhs.getSemantics(), llvm::RoundingMode::TowardZero, &losesInfo);
  return result;
}

} // namespace mli

#endif // MLI_UTILS_H
