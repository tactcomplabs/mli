//===- LLVMInterpreter.cpp - LLVM dialect interpreter -------------*- C++ -*-===//
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
#include "mlir/Interpreter/Dialects/LLVMInterpreter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
typedef llvm::APFloat::Semantics Semantics;

APInt getIntegerData(const EvalValue val, const bool isSigned = false) {
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
Semantics getFloatSemantics(const Type result_type) {
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

APFloat getFloatData(const EvalValue val) {
    if (val.getType().isF64()) {
        double num = val.getData<double>().front();
        return APFloat(num);
    }
    float num = val.getData<float>().front();
    return APFloat(num);
}

// The APFloat library doesn't support standard math functions (sqrt, exp, etc)
// Convert num to double, compute function, and store result back into APFloat
APFloat compute(const APFloat& num, double (*func)(double)) {
    double tmp = num.convertToDouble();
    tmp = func(tmp);
    APFloat result = APFloat(tmp);
    bool losesInfo;
    // Copy semantics from original num to ensure that dtype is the same
    result.convert(num.getSemantics(), llvm::RoundingMode::TowardZero, &losesInfo);
    return result;
}

namespace {

class LLVMReturnOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMReturnOpInterpreter,
                                                   LLVM::ReturnOp> {
public:
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    return interpreter.createReturnValueResult(operands);
  }
};

struct LLVMFNegOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFNegOpInterpreter,
                                                   LLVM::FNegOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = -lhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMAddOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAddOpInterpreter, LLVM::AddOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs + rhs;
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";

    // Create an EvalValue from the result
    // NOTE: Is the result's MLIR type from getType() compatible with APInt/APFloat?
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));

    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFAddOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFAddOpInterpreter,
                                                   LLVM::FAddOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat rhs = getFloatData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ");
    APFloat result = lhs + rhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSubOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSubOpInterpreter,
                                                   LLVM::SubOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs - rhs;
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFSubOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFSubOpInterpreter,
                                                   LLVM::FSubOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat rhs = getFloatData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ");
    APFloat result = lhs - rhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMMulOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMMulOpInterpreter,
                                                   LLVM::MulOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs * rhs;
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFMulOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFMulOpInterpreter,
                                                   LLVM::FMulOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat rhs = getFloatData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ");
    APFloat result = lhs * rhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMUDivOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMUDivOpInterpreter,
                                                   LLVM::UDivOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs.udiv(rhs);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSDivOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSDivOpInterpreter,
                                                   LLVM::SDivOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs.sdiv(rhs);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFDivOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFDivOpInterpreter,
                                                   LLVM::FDivOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat rhs = getFloatData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ");
    APFloat result = lhs / rhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMURemOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMURemOpInterpreter,
                                                   LLVM::URemOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs.urem(rhs);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSRemOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSRemOpInterpreter,
                                                   LLVM::SRemOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs.srem(rhs);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFRemOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFRemOpInterpreter,
                                                   LLVM::FRemOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat rhs = getFloatData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ");

    lhs.mod(rhs);
    APFloat result = lhs;

    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMShlOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMShlOpInterpreter,
                                                   LLVM::ShlOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs.shl(rhs);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMLShrOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMLShrOpInterpreter,
                                                   LLVM::LShrOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs.lshr(rhs);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMAShrOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAShrOpInterpreter,
                                                   LLVM::AShrOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs.ashr(rhs);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMAndOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAndOpInterpreter,
                                                   LLVM::AndOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs & rhs;
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMOrOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMOrOpInterpreter,
                                                   LLVM::OrOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs | rhs;
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMXOrOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMXOrOpInterpreter,
                                                   LLVM::XOrOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt rhs = getIntegerData(operands[1]);
    rhs.print(llvm::outs() << "\nrhs: ", isSigned);
    APInt result = lhs ^ rhs;
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMTruncOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMTruncOpInterpreter,
                                                   LLVM::TruncOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    unsigned result_width = op->getOpResult(0).getType().getIntOrFloatBitWidth();
    APInt result = lhs.trunc(result_width);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFPTruncOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFPTruncOpInterpreter,
                                                   LLVM::FPTruncOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");

    bool loseInfo;
    auto result_type = op->getOpResult(0).getType();
    lhs.convert(llvm::APFloatBase::EnumToSemantics(getFloatSemantics(result_type)), llvm::APFloat::rmTowardZero, &loseInfo);

    APFloat result = lhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMZExtOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMZExtOpInterpreter,
                                                   LLVM::ZExtOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true; // garbage value
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    lhs.print(llvm::outs() << "lhs: ", isSigned);

    unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    APInt result = lhs.zext(result_width);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSExtOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSExtOpInterpreter,
                                                   LLVM::SExtOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    lhs.print(llvm::outs() << "lhs: ", isSigned);

    unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    llvm::outs() << "Extending to " << result_width << " bits\n";
    APInt result = lhs.sext(result_width);
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMBitcastOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMBitcastOpInterpreter,
                                                   LLVM::BitcastOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    auto src_type = operands[0].getType();
    auto result_type = op->getOpResult(0).getType();
    EvalValue evalResult;

    if (src_type == result_type) { // no-op
        evalResult = operands[0];
    }
    else if (!src_type.isInteger() && src_type.isIntOrFloat()) { // src float, ret int
        APFloat lhs = getFloatData(operands[0]);
        // FIXME: This bitcast always gives zero
        APInt result = lhs.bitcastToAPInt();
        result.print(llvm::outs() << "\nresult: ", true);
        evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
    }
    else { // src int, ret float
        APInt lhs = getIntegerData(operands[0]);
        Semantics s = getFloatSemantics(result_type);
        APFloat result = APFloat(llvm::APFloat::EnumToSemantics(s), lhs);
        result.print(llvm::outs() << "\nresult: ");
        llvm::outs() << "\n";
        evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
    }
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFPExtOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFPExtOpInterpreter,
                                                   LLVM::FPExtOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");

    bool loseInfo;
    auto result_type = op->getOpResult(0).getType();
    lhs.convert(llvm::APFloatBase::EnumToSemantics(getFloatSemantics(result_type)), llvm::APFloat::rmTowardZero, &loseInfo);

    APFloat result = lhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFPToSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFPToSIOpInterpreter,
                                                   LLVM::FPToSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");

    unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    bool isUnsigned = false;
    APSInt result = APSInt(result_width, isUnsigned);
    bool isExact;
    lhs.convertToInteger(result, llvm::RoundingMode::TowardZero, &isExact);
    llvm::outs() << "result: " << result << "\n";

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFPToUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFPToUIOpInterpreter,
                                                   LLVM::FPToUIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");

    unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    bool isUnsigned = true;
    APSInt result = APSInt(result_width, isUnsigned);
    bool isExact;
    lhs.convertToInteger(result, llvm::RoundingMode::TowardZero, &isExact);
    llvm::outs() << "result: " << result << "\n";

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSIToFPOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSIToFPOpInterpreter,
                                                   LLVM::SIToFPOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    lhs.print(llvm::outs() << "lhs: ", isSigned);

    Semantics sem = getFloatSemantics(op->getResult(0).getType());
    APFloat result = APFloat(llvm::APFloatBase::EnumToSemantics(sem));
    result.convertFromAPInt(lhs, isSigned, llvm::RoundingMode::TowardZero);

    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMUIToFPOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMUIToFPOpInterpreter,
                                                   LLVM::UIToFPOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    lhs.print(llvm::outs() << "lhs: ", isSigned);

    Semantics sem = getFloatSemantics(op->getResult(0).getType());
    APFloat result = APFloat(llvm::APFloatBase::EnumToSemantics(sem));
    result.convertFromAPInt(lhs, isSigned, llvm::RoundingMode::TowardZero);

    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMAbsOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAbsOpInterpreter,
                                                   LLVM::AbsOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt result = lhs.abs();
    result.print(llvm::outs() << "\nresult: ", isSigned);
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMCosOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMCosOpInterpreter,
                                                   LLVM::CosOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");

    APFloat result = compute(lhs, std::cos);
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMExp2OpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMExp2OpInterpreter,
                                                   LLVM::Exp2Op> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = compute(lhs, std::exp2);
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMExpOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMExpOpInterpreter,
                                                   LLVM::ExpOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = compute(lhs, std::exp);
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFAbsOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFAbsOpInterpreter,
                                                   LLVM::FAbsOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = compute(lhs, std::fabs);
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFCeilOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFCeilOpInterpreter,
                                                   LLVM::FCeilOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");

    lhs.roundToIntegral(llvm::RoundingMode::TowardPositive);
    APFloat result = lhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFFloorOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFFloorOpInterpreter,
                                                   LLVM::FFloorOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");

    lhs.roundToIntegral(llvm::RoundingMode::TowardNegative);
    APFloat result = lhs;
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFMAOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFMAOpInterpreter,
                                                   LLVM::FMAOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = lhs; // TODO:
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMLog10OpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMLog10OpInterpreter,
                                                   LLVM::Log10Op> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = compute(lhs, std::log10);
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMLog2OpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMLog2OpInterpreter,
                                                   LLVM::Log2Op> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = compute(lhs, std::log2);
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMLogOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMLogOpInterpreter,
                                                   LLVM::LogOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = compute(lhs, std::log);
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSinOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSinOpInterpreter,
                                                   LLVM::SinOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");
    APFloat result = compute(lhs, std::sin);
    result.print(llvm::outs() << "\nresult: ");
    llvm::outs() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

// addressof
//
// Creates a pointer pointing to a global or a function
// Syntax:
// operation ::= `llvm.mlir.addressof` $global_name attr-dict `:` qualified(type($res))
struct LLVMAddressOfOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAddressOfOpInterpreter,
                                                   LLVM::AddressOfOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    auto globalName = op->getAttrOfType<StringAttr>("global_name");
    llvm::outs() << "global_name: " << globalName.getValue() << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &globalName, sizeof(globalName));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

// alloca
//
// Interfaces: DestructurableAllocationOpInterface, GetResultPtrElementType, PromotableAllocationOpInterface

// alignment	::mlir::IntegerAttr	64-bit signless integer attribute
// elem_type	::mlir::TypeAttr	any type attribute
// inalloca	::mlir::UnitAttr	unit attribute

// Operands:
// arraySize: signless integer
// Result:	LLVM pointer type

struct LLVMAllocaOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAllocaOpInterpreter,
                                                   LLVM::AllocaOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting " << op->getName() << "\n";
    auto arraySize = operands[0].getData<int>().front();
    llvm::outs() << "arraySize: " << arraySize << "\n";
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &arraySize, sizeof(arraySize));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

} // namespace

void LLVMInterpreter::attachInterface(MLIRContext &context) {
    // Arithmetic
    LLVM::FNegOp::attachInterface<LLVMFNegOpInterpreter>(context);
    LLVM::AddOp::attachInterface<LLVMAddOpInterpreter>(context);
    LLVM::FAddOp::attachInterface<LLVMFAddOpInterpreter>(context);
    LLVM::SubOp::attachInterface<LLVMSubOpInterpreter>(context);
    LLVM::FSubOp::attachInterface<LLVMFSubOpInterpreter>(context);
    LLVM::MulOp::attachInterface<LLVMMulOpInterpreter>(context);
    LLVM::FMulOp::attachInterface<LLVMFMulOpInterpreter>(context);
    LLVM::SDivOp::attachInterface<LLVMSDivOpInterpreter>(context);
    LLVM::UDivOp::attachInterface<LLVMUDivOpInterpreter>(context);
    LLVM::FDivOp::attachInterface<LLVMFDivOpInterpreter>(context);
    LLVM::SRemOp::attachInterface<LLVMSRemOpInterpreter>(context);
    LLVM::URemOp::attachInterface<LLVMURemOpInterpreter>(context);
    LLVM::FRemOp::attachInterface<LLVMFRemOpInterpreter>(context);

    // Bitwise
    LLVM::ShlOp::attachInterface<LLVMShlOpInterpreter>(context);
    LLVM::LShrOp::attachInterface<LLVMLShrOpInterpreter>(context);
    LLVM::AShrOp::attachInterface<LLVMAShrOpInterpreter>(context);
    LLVM::AndOp::attachInterface<LLVMAndOpInterpreter>(context);
    LLVM::OrOp::attachInterface<LLVMOrOpInterpreter>(context);
    LLVM::XOrOp::attachInterface<LLVMXOrOpInterpreter>(context);

    // Memory
    LLVM::ReturnOp::attachInterface<LLVMReturnOpInterpreter>(context);
    LLVM::AddressOfOp::attachInterface<LLVMAddressOfOpInterpreter>(context);

    // Conversions
    LLVM::TruncOp::attachInterface<LLVMTruncOpInterpreter>(context);
    LLVM::ZExtOp::attachInterface<LLVMZExtOpInterpreter>(context);
    LLVM::SExtOp::attachInterface<LLVMSExtOpInterpreter>(context);
    LLVM::FPTruncOp::attachInterface<LLVMFPTruncOpInterpreter>(context);
    LLVM::FPExtOp::attachInterface<LLVMFPExtOpInterpreter>(context);
    LLVM::BitcastOp::attachInterface<LLVMBitcastOpInterpreter>(context);
    LLVM::FPToSIOp::attachInterface<LLVMFPToSIOpInterpreter>(context);
    LLVM::FPToUIOp::attachInterface<LLVMFPToUIOpInterpreter>(context);
    LLVM::SIToFPOp::attachInterface<LLVMSIToFPOpInterpreter>(context);
    LLVM::UIToFPOp::attachInterface<LLVMUIToFPOpInterpreter>(context);

    // Math Intrinsics
    LLVM::AbsOp::attachInterface<LLVMAbsOpInterpreter>(context);
    LLVM::CosOp::attachInterface<LLVMCosOpInterpreter>(context);
    // ::math::CoshOp::attachInterface<LLVMCoshOpInterpreter>(context);
    LLVM::Exp2Op::attachInterface<LLVMExp2OpInterpreter>(context);
    LLVM::ExpOp::attachInterface<LLVMExpOpInterpreter>(context);
    LLVM::FAbsOp::attachInterface<LLVMFAbsOpInterpreter>(context);
    LLVM::FCeilOp::attachInterface<LLVMFCeilOpInterpreter>(context);
    LLVM::FFloorOp::attachInterface<LLVMFFloorOpInterpreter>(context);
    LLVM::FMAOp::attachInterface<LLVMFMAOpInterpreter>(context);
    LLVM::Log10Op::attachInterface<LLVMLog10OpInterpreter>(context);
    LLVM::Log2Op::attachInterface<LLVMLog2OpInterpreter>(context);
    LLVM::LogOp::attachInterface<LLVMLogOpInterpreter>(context);
    LLVM::SinOp::attachInterface<LLVMSinOpInterpreter>(context);
    /*
    // ::math::SinhOp::attachInterface<LLVMSinhOpInterpreter>(context);
    LLVM::SMaxOp::attachInterface<LLVMSMaxOpInterpreter>(context);
    LLVM::SMinOp::attachInterface<LLVMSMinOpInterpreter>(context);
    LLVM::SqrtOp::attachInterface<LLVMSqrtOpInterpreter>(context);
    // LLVM::TanOp::attachInterface<LLVMTanOpInterpreter>(context);
    // LLVM::TanhOp::attachInterface<LLVMTanhOpInterpreter>(context);
    LLVM::UMaxOp::attachInterface<LLVMUMaxOpInterpreter>(context);
    LLVM::UMinOp::attachInterface<LLVMUMinOpInterpreter>(context);

    // Bit Manipulation Intrinsics
    LLVM::BitReverseOp::attachInterface<LLVMBitReverseOpInterpreter>(context);
    LLVM::ByteSwapOp::attachInterface<LLVMByteSwapOpInterpreter>(context);
    LLVM::CountLeadingZerosOp::attachInterface<LLVMCountLeadingZerosOpInterpreter>(context);
    LLVM::CountTrailingZerosOp::attachInterface<LLVMCountTrailingZerosOpInterpreter>(context);
    LLVM::CtPopOp::attachInterface<LLVMCtPopOpInterpreter>(context);
    LLVM::FShlOp::attachInterface<LLVMFShlOpInterpreter>(context);
    LLVM::FShrOp::attachInterface<LLVMFShrOpInterpreter>(context);
    */

}
