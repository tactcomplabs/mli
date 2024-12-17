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
#include "llvm/ADT/APFloat.h"
#include <cmath>

using namespace mlir;

APInt getIntegerData(const EvalValue val, const bool isSigned = false) {
    uint64_t num = val.getData<uint64_t>().front();
    size_t width = val.getRawDataSizeInBytes();
    return APInt(width, num, isSigned);
}

APFloat getFloatData(const EvalValue val) {
    double num = val.getData<double>().front();
    size_t width = val.getRawDataSizeInBytes();
    if (width == 8 * sizeof(float)) {
        return APFloat((float)num);
    }
    return APFloat(num);
}

namespace {

class LLVMReturnOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMReturnOpInterpreter,
                                                   LLVM::ReturnOp> {
public:
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::ReturnOp\n";
    return interpreter.createReturnValueResult(operands);
  }
};

struct LLVMFNegOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFNegOpInterpreter,
                                                   LLVM::FNegOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::FNeg\n";
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
  llvm::outs() << "Interpreting LLVM::AddOp\n";
  APInt lhs = getIntegerData(operands[0]);  
  lhs.print(llvm::outs() << "lhs: ", isSigned);
  APInt rhs = getIntegerData(operands[1]);
  rhs.print(llvm::outs() << "\nrhs: ", isSigned);
  APInt result = lhs + rhs;
  result.print(llvm::outs() << "\nresult: ", isSigned);
  llvm::outs() << "\n";

  // Create an EvalValue from the result
  auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));

  // Wrap the EvalValue in an ArrayRef and return the EvalResult
    //return interpreter.createReturnValueResult({evalResult});
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFAddOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFAddOpInterpreter,
                                                   LLVM::FAddOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::FAdd\n";
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
    llvm::outs() << "Interpreting LLVM::Sub\n";
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
    llvm::outs() << "Interpreting LLVM::FSub\n";
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
    llvm::outs() << "Interpreting LLVM::Mul\n";
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
    llvm::outs() << "Interpreting LLVM::FMul\n";
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
    llvm::outs() << "Interpreting LLVM::UDiv\n";
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
    llvm::outs() << "Interpreting LLVM::SDiv\n";
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
    llvm::outs() << "Interpreting LLVM::FDiv\n";
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
    llvm::outs() << "Interpreting LLVM::URem\n";
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
    llvm::outs() << "Interpreting LLVM::SRem\n";
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
    llvm::outs() << "Interpreting LLVM::FRem\n";
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
    llvm::outs() << "Interpreting LLVM::Shl\n";
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
    llvm::outs() << "Interpreting LLVM::LShr\n";
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
    llvm::outs() << "Interpreting LLVM::AShr\n";
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
    llvm::outs() << "Interpreting LLVM::And\n";
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
    llvm::outs() << "Interpreting LLVM::Or\n";
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
    llvm::outs() << "Interpreting LLVM::XOr\n";
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
    // bool isSigned = true;
    llvm::outs() << "Interpreting LLVM::Trunc\n";
    llvm::errs() << "UNIMPLEMENTED\n";
    APInt lhs = getIntegerData(operands[0]);
    //lhs.print(llvm::outs() << "lhs: ", isSigned);
    APInt result = lhs;
    // result.print(llvm::outs() << "\nresult: ", isSigned);
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
    llvm::outs() << "Interpreting LLVM::AddressOfOp\n";
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
    llvm::outs() << "Interpreting LLVM::AllocaOp\n";
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
    LLVM::ReturnOp::attachInterface<LLVMReturnOpInterpreter>(context);
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
    LLVM::ShlOp::attachInterface<LLVMShlOpInterpreter>(context);
    LLVM::LShrOp::attachInterface<LLVMLShrOpInterpreter>(context);
    LLVM::AShrOp::attachInterface<LLVMAShrOpInterpreter>(context);
    LLVM::AndOp::attachInterface<LLVMAndOpInterpreter>(context);
    LLVM::OrOp::attachInterface<LLVMOrOpInterpreter>(context);
    LLVM::XOrOp::attachInterface<LLVMXOrOpInterpreter>(context);
    LLVM::AddressOfOp::attachInterface<LLVMAddressOfOpInterpreter>(context);
    LLVM::TruncOp::attachInterface<LLVMTruncOpInterpreter>(context);
}
