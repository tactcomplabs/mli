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

using namespace mlir;

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

struct LLVMAddOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAddOpInterpreter, LLVM::AddOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
  llvm::outs() << "Interpreting LLVM::AddOp\n";
  int lhs = operands[0].getData<int>().front();
  llvm::outs() << "lhs: " << lhs << "\n";
  int rhs = operands[1].getData<int>().front();
  llvm::outs() << "rhs: " << rhs << "\n";
  int result = lhs + rhs;
  llvm::outs() << "result: " << result << "\n";

  // Create an EvalValue from the result
  auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));

  // Wrap the EvalValue in an ArrayRef and return the EvalResult
    //return interpreter.createReturnValueResult({evalResult});
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSubOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSubOpInterpreter, LLVM::SubOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
  llvm::outs() << "Interpreting LLVM::SubOp\n";
  int lhs = operands[0].getData<int>().front();
  llvm::outs() << "lhs: " << lhs << "\n";
  int rhs = operands[1].getData<int>().front();
  llvm::outs() << "rhs: " << rhs << "\n";
  int result = lhs - rhs;
  llvm::outs() << "result: " << result << "\n";

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
    llvm::outs() << "Interpreting LLVM::ASHROp\n";
    int lhs = operands[0].getData<int>().front();
    llvm::outs() << "lhs: " << lhs << "\n";
    int rhs = operands[1].getData<int>().front();
    llvm::outs() << "rhs: " << rhs << "\n";
    int result = lhs >> rhs;
    llvm::outs() << "result: " << result << "\n";
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
    llvm::outs() << "Interpreting LLVM::SHLOp\n";
    int lhs = operands[0].getData<int>().front();
    llvm::outs() << "lhs: " << lhs << "\n";
    int rhs = operands[1].getData<int>().front();
    llvm::outs() << "rhs: " << rhs << "\n";
    int result = lhs << rhs;
    llvm::outs() << "result: " << result << "\n";
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
    llvm::outs() << "Interpreting LLVM::ANDOp\n";
    int lhs = operands[0].getData<int>().front();
    llvm::outs() << "lhs: " << lhs << "\n";
    int rhs = operands[1].getData<int>().front();
    llvm::outs() << "rhs: " << rhs << "\n";
    int result = lhs & rhs;
    llvm::outs() << "result: " << result << "\n";
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
    llvm::outs() << "Interpreting LLVM::OROp\n";
    int lhs = operands[0].getData<int>().front();
    llvm::outs() << "lhs: " << lhs << "\n";
    int rhs = operands[1].getData<int>().front();
    llvm::outs() << "rhs: " << rhs << "\n";
    int result = lhs | rhs;
    llvm::outs() << "result: " << result << "\n";
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
    llvm::outs() << "Interpreting LLVM::MULOp\n";
    int lhs = operands[0].getData<int>().front();
    llvm::outs() << "lhs: " << lhs << "\n";
    int rhs = operands[1].getData<int>().front();
    llvm::outs() << "rhs: " << rhs << "\n";
    int result = lhs * rhs;
    llvm::outs() << "result: " << result << "\n";
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
    llvm::outs() << "Interpreting LLVM::XOROp\n";
    int lhs = operands[0].getData<int>().front();
    llvm::outs() << "lhs: " << lhs << "\n";
    int rhs = operands[1].getData<int>().front();
    llvm::outs() << "rhs: " << rhs << "\n";
    int result = lhs ^ rhs;
    llvm::outs() << "result: " << result << "\n";
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
  LLVM::SubOp::attachInterface<LLVMSubOpInterpreter>(context);
  LLVM::AddOp::attachInterface<LLVMAddOpInterpreter>(context);
  LLVM::ReturnOp::attachInterface<LLVMReturnOpInterpreter>(context);
  LLVM::AShrOp::attachInterface<LLVMAShrOpInterpreter>(context);
  LLVM::ShlOp::attachInterface<LLVMShlOpInterpreter>(context);
  LLVM::AndOp::attachInterface<LLVMAndOpInterpreter>(context);
  LLVM::OrOp::attachInterface<LLVMOrOpInterpreter>(context);
  LLVM::MulOp::attachInterface<LLVMMulOpInterpreter>(context);
  LLVM::XOrOp::attachInterface<LLVMXOrOpInterpreter>(context);
  LLVM::AddressOfOp::attachInterface<LLVMAddressOfOpInterpreter>(context);

}
