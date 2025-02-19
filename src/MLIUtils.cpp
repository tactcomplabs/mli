//===- MLIUtils.cpp - Multi-Level Interpreter Utilities -------------*- C++ -*-===//
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

#include "MLIUtils.h"
#include "MLIFormat.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

typedef llvm::APFloatBase::Semantics Semantics;
using namespace llvm;

namespace mli {

mlir::OwningOpRef<mlir::ModuleOp> parseMLIRFile(llvm::StringRef filename,
                                                mlir::MLIRContext &context) {
  // Open the input file
  std::string errorMessage;
  auto file = mlir::openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  // Create a source manager for the input file
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // Create parser config
  mlir::ParserConfig config(&context);

  // Attempt to parse the file (text or bytecode)
  if (auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, config)) {
    llvm::outs() << "Successfully parsed MLIR file: " << filename << "\n";
    return module;
  }

  llvm::errs() << "Failed to parse MLIR file (unsupported format)\n";
  return nullptr;
}

void printFunctionSignature(mlir::func::FuncOp funcOp, llvm::raw_ostream &os) {
  os << mli::fmt::dim("Function '") << mli::fmt::highlight(funcOp.getName().str()) << mli::fmt::dim("' signature:") << "\n";
  os << "  " << mli::fmt::highlight(funcOp.getName().str()) << "(";
  
  mlir::FunctionType fnType = funcOp.getFunctionType();
  for (size_t i = 0; i < fnType.getNumInputs(); ++i) {
    if (i > 0) os << ", ";
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
      if (i > 0) os << ", ";
      std::string typeStr;
      llvm::raw_string_ostream typeOs(typeStr);
      fnType.getResult(i).print(typeOs);
      os << mli::fmt::type(typeStr);
    }
  }
  os << "\n";
}

bool validateFunctionArguments(mlir::func::FuncOp funcOp, 
                             llvm::ArrayRef<int32_t> providedArgs,
                             std::string &errorMessage) {
  // Get function type
  mlir::FunctionType fnType = funcOp.getFunctionType();
  
  std::string signature;
  llvm::raw_string_ostream os(signature);
  printFunctionSignature(funcOp, os);
  
  // Check number of arguments
  if (fnType.getNumInputs() != providedArgs.size()) {
    errorMessage = mli::fmt::error("Argument count mismatch") + "\n" +
    signature +
    mli::fmt::dim("Provided ") + mli::fmt::highlight(std::to_string(providedArgs.size())) + 
    mli::fmt::dim(" argument(s): [");

    // Print provided arguments with their inferred type
    for (size_t i = 0; i < providedArgs.size(); ++i) {
      if (i > 0){
          errorMessage += ", ";
      }
      errorMessage += mli::fmt::highlight(std::to_string(providedArgs[i]));
    }
    errorMessage += mli::fmt::dim("] -> ") + mli::fmt::type("i32") + "\n" +
    mli::fmt::dim("Expected ") + mli::fmt::highlight(std::to_string(fnType.getNumInputs())) + 
    mli::fmt::dim(" argument(s)");
    return false;
  }

  // Check argument types
  for (size_t i = 0; i < fnType.getNumInputs(); ++i) {
    mlir::Type argType = fnType.getInput(i);
    // TODO: implement floating point types 
    if (!mlir::isa<mlir::IntegerType>(argType)) {
      errorMessage = mli::fmt::error("Type mismatch for argument " + std::to_string(i)) + "\n" +
        signature +
        mli::fmt::dim("Provided value: ") + mli::fmt::highlight(std::to_string(providedArgs[i])) + 
        mli::fmt::dim(" -> ") + mli::fmt::type("i32") + "\n" +
        mli::fmt::dim("Expected type: ") + mli::fmt::type(std::to_string(mlir::cast<mlir::IntegerType>(argType).getWidth()) + "-bit integer");
      return false;
    }

    // Check integer width matches our int32_t
    auto intType = mlir::cast<mlir::IntegerType>(argType);
    if (intType.getWidth() != 32) {
      errorMessage = mli::fmt::error("Integer width mismatch for argument " + std::to_string(i)) + "\n" +
                  signature +
                  mli::fmt::dim("Provided value: ") + mli::fmt::highlight(std::to_string(providedArgs[i])) + 
                  mli::fmt::dim(" -> ") + mli::fmt::type("i32") + "\n" +
                  mli::fmt::dim("Expected: ") + mli::fmt::type(std::to_string(intType.getWidth()) + "-bit integer");
      return false;
    }
  }
  return true;
}


// --- APInt and APFloat utilities ---
APInt getIntegerData(const mlir::EvalValue val, const bool isSigned) {
    // NOTE: Is this a bad access if sizeof(val) < sizeof(num)?
    // When constructing val from the CLI args, we supply its width
    // This is deduced from the type of the function's arguments
    // Calling getData with uint64_t could read more memory than actually allocated for val
    uint64_t num = val.getData<uint64_t>().front();
    size_t width = val.getRawDataSizeInBytes();
    return APInt(width, num, isSigned);
}

// Get semantics for given MLIR Type
// These are used in constructing/converting APFloat types
Semantics getFloatSemantics(const mlir::Type result_type) {
    if (result_type.isF16()) { // 16 bit float
        return Semantics::S_IEEEhalf;
    }
    else if (result_type.isBF16()) { // 16 bit brain float
        return Semantics::S_BFloat;
    }
    else if (result_type.isF32()) { // standard 32 bit float
        return Semantics::S_IEEEsingle;
    }
    return Semantics::S_IEEEdouble;
}

APFloat getFloatData(const mlir::EvalValue val) {
    if (val.getType().isF64()) {
        double num = val.getData<double>().front();
        return APFloat(num);
    }
    float num = val.getData<float>().front();
    return APFloat(num);
}

// The APFloat library doesn't support standard math functions
// Convert num to double, compute function, and store result back into APFloat
// For unary operations (std::sqrt, std::cos, etc.)
APFloat compute(const APFloat& num, double (*func)(double)) {
    double tmp = num.convertToDouble();
    tmp = func(tmp);
    APFloat result = APFloat(tmp);
    bool losesInfo;
    // Copy semantics from original num to ensure that dtype is the same
    result.convert(num.getSemantics(), llvm::RoundingMode::TowardZero, &losesInfo);
    return result;
}

// For binary operations on two floats (std::pow, std::fmax)
APFloat compute(const APFloat& lhs, const APFloat& rhs, double (*func)(double, double)) {
    double tmp1 = lhs.convertToDouble();
    double tmp2 = rhs.convertToDouble();
    tmp1 = func(tmp1, tmp2);
    APFloat result = APFloat(tmp1);
    bool losesInfo;
    // NOTE: Is this a good idea? What if signature is (f32, f32) -> f64?
    result.convert(lhs.getSemantics(), llvm::RoundingMode::TowardZero, &losesInfo);
    return result;
}

// For binary operations on float and int (std::powi)
APFloat compute(const APFloat& lhs, const APInt& rhs, double (*func)(double, int64_t)) {
    double tmp1 = lhs.convertToDouble();
    int64_t tmp2 = rhs.getSExtValue();
    tmp1 = func(tmp1, tmp2);
    APFloat result = APFloat(tmp1);
    bool losesInfo;
    // NOTE: Is this a good idea? What if signature is (f32, f32) -> f64?
    result.convert(lhs.getSemantics(), llvm::RoundingMode::TowardZero, &losesInfo);
    return result;
}


} // namespace mli
