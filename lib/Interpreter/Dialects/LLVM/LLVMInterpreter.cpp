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
#include "MLIFormat.h"
#include "MLIUtils.h"
#include "mlir/Interpreter/Dialects/LLVMInterpreter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
using namespace mli;
typedef llvm::APFloat::Semantics Semantics;

namespace {

class LLVMReturnOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMReturnOpInterpreter,
                                                   LLVM::ReturnOp> {
public:
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << mli::fmt::info("Interpreting LLVM::ReturnOp") << "\n";
    return interpreter.createReturnValueResult(operands);
  }
};

struct LLVMFNegOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFNegOpInterpreter,
                                                   LLVM::FNegOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = -lhs;
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs + rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);

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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = lhs + rhs;
    mli::fmt::printResult(llvm::outs(), result);

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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());

    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs - rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFSubOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFSubOpInterpreter,
                                                   LLVM::FSubOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = lhs - rhs;
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMMulOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMMulOpInterpreter, LLVM::MulOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs * rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);

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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = lhs * rhs;
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.udiv(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    rhs.print(llvm::outs() << "rhs: ", isSigned);
    APInt result = lhs.sdiv(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = lhs / rhs;
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.urem(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.srem(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    lhs.mod(rhs);
    APFloat result = lhs;

    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMICmpOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMICmpOpInterpreter,
                                                   LLVM::ICmpOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    // icmp has various attributes specifying the type of comparison to make
    LLVM::ICmpPredicate att = op->getAttrOfType<LLVM::ICmpPredicateAttr>("predicate").getValue();
    bool isSigned = att == LLVM::ICmpPredicate::slt || att == LLVM::ICmpPredicate::sle
                 || att == LLVM::ICmpPredicate::sgt || att == LLVM::ICmpPredicate::sge;

    llvm::outs() << "Interpreting LLVM::ICmp\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);

    bool result;
    llvm::outs() << "attribute: " << att << '\n';
    switch (att) {
        case LLVM::ICmpPredicate::ne:
            result = lhs.ne(rhs);
            break;
        case LLVM::ICmpPredicate::slt:
            result = lhs.slt(rhs);
            break;
        case LLVM::ICmpPredicate::sle:
            result = lhs.sle(rhs);
            break;
        case LLVM::ICmpPredicate::sgt:
            result = lhs.sgt(rhs);
            break;
        case LLVM::ICmpPredicate::sge:
            result = lhs.sge(rhs);
            break;
        case LLVM::ICmpPredicate::ult:
            result = lhs.ult(rhs);
            break;
        case LLVM::ICmpPredicate::ule:
            result = lhs.ule(rhs);
            break;
        case LLVM::ICmpPredicate::ugt:
            result = lhs.ugt(rhs);
            break;
        case LLVM::ICmpPredicate::uge:
            result = lhs.uge(rhs);
            break;
        default:
            result = lhs.eq(rhs);
    }

    mli::fmt::printResult(llvm::outs(), result);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMFCmpOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMFCmpOpInterpreter,
                                                   LLVM::FCmpOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    // FCmp has various attributes specifying the type of comparison to make
    LLVM::FCmpPredicate att = op->getAttrOfType<LLVM::FCmpPredicateAttr>("predicate").getValue();
    llvm::outs() << "Interpreting LLVM::FCmp\n";
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    bool lhs_NaN = lhs.isNaN();
    bool rhs_NaN = rhs.isNaN();
    bool result;
    // FCmp supports both ordered and unordered comparisons
    // Any ordered comparison returns false if at least one argument is NaN
    // Any unordered comparison returns true if at least one argument is NaN
    switch (att) {
        case LLVM::FCmpPredicate::_false:
            result = false;
            break;
        case LLVM::FCmpPredicate::oeq:
            result = !(lhs_NaN || rhs_NaN) && lhs == rhs;
            break;
        case LLVM::FCmpPredicate::ogt:
            result = !(lhs_NaN || rhs_NaN) && lhs > rhs;
            break;
        case LLVM::FCmpPredicate::oge:
            result = !(lhs_NaN || rhs_NaN) && lhs >= rhs;
            break;
        case LLVM::FCmpPredicate::olt:
            result = !(lhs_NaN || rhs_NaN) && lhs < rhs;
            break;
        case LLVM::FCmpPredicate::one:
            result = !(lhs_NaN || rhs_NaN) && lhs != rhs;
            break;
        case LLVM::FCmpPredicate::ord:
            result = !(lhs_NaN || rhs_NaN);
            break;
        case LLVM::FCmpPredicate::ueq:
            result = lhs_NaN || rhs_NaN || lhs == rhs;
            break;
        case LLVM::FCmpPredicate::ugt:
            result = lhs_NaN || rhs_NaN || lhs > rhs;
            break;
        case LLVM::FCmpPredicate::uge:
            result = lhs_NaN || rhs_NaN || lhs >= rhs;
            break;
        case LLVM::FCmpPredicate::ult:
            result = lhs_NaN || rhs_NaN || lhs < rhs;
            break;
        case LLVM::FCmpPredicate::ule:
            result = lhs_NaN || rhs_NaN || lhs <= rhs;
            break;
        case LLVM::FCmpPredicate::une:
            result = lhs_NaN || rhs_NaN || lhs != rhs;
            break;
        case LLVM::FCmpPredicate::uno:
            result = lhs_NaN || rhs_NaN;
            break;
        default:
            result = true;
    }

    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.shl(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.lshr(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);

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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.ashr(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs & rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs | rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMXOrOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMXOrOpInterpreter, LLVM::XOrOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs ^ rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    unsigned result_width = op->getOpResult(0).getType().getIntOrFloatBitWidth();
    APInt result = lhs.trunc(result_width);
    mli::fmt::printResult(llvm::outs(), result, isSigned);

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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);

    bool loseInfo;
    auto result_type = op->getOpResult(0).getType();
    lhs.convert(llvm::APFloatBase::EnumToSemantics(getFloatSemantics(result_type)), llvm::APFloat::rmTowardZero, &loseInfo);

    APFloat result = lhs;
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);

    unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    APInt result = lhs.zext(result_width);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);

    unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    llvm::outs() << "Extending to " << result_width << " bits\n";
    APInt result = lhs.sext(result_width);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    auto src_type = operands[0].getType();
    auto result_type = op->getOpResult(0).getType();
    EvalValue evalResult;

    if (src_type == result_type) { // no-op
        evalResult = operands[0];
    }
    else if (!src_type.isInteger() && src_type.isIntOrFloat()) { // src float, ret int
        APFloat lhs = getFloatData(operands[0]);
        mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
        APInt result = lhs.bitcastToAPInt();
        mli::fmt::printResult(llvm::outs(), result);
        evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
    }
    else { // src int, ret float
        APInt lhs = getIntegerData(operands[0]);
        mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
        Semantics s = getFloatSemantics(result_type);
        APFloat result = APFloat(llvm::APFloat::EnumToSemantics(s), lhs);
        mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);

    bool loseInfo;
    auto result_type = op->getOpResult(0).getType();
    lhs.convert(llvm::APFloatBase::EnumToSemantics(getFloatSemantics(result_type)), llvm::APFloat::rmTowardZero, &loseInfo);

    APFloat result = lhs;
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);

    unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    bool isUnsigned = false;
    APSInt result = APSInt(result_width, isUnsigned);
    bool isExact;
    lhs.convertToInteger(result, llvm::RoundingMode::TowardZero, &isExact);
    mli::fmt::printResult(llvm::outs(), result);

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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    lhs.print(llvm::outs() << "lhs: ");

    unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    bool isUnsigned = true;
    APSInt result = APSInt(result_width, isUnsigned);
    bool isExact;
    lhs.convertToInteger(result, llvm::RoundingMode::TowardZero, &isExact);
    mli::fmt::printResult(llvm::outs(), result);

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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);

    Semantics sem = getFloatSemantics(op->getResult(0).getType());
    APFloat result = APFloat(llvm::APFloatBase::EnumToSemantics(sem));
    result.convertFromAPInt(lhs, isSigned, llvm::RoundingMode::TowardZero);

    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);

    Semantics sem = getFloatSemantics(op->getResult(0).getType());
    APFloat result = APFloat(llvm::APFloatBase::EnumToSemantics(sem));
    result.convertFromAPInt(lhs, isSigned, llvm::RoundingMode::TowardZero);

    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt result = lhs.abs();
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);

    APFloat result = compute(lhs, std::cos);
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = compute(lhs, std::exp2);
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = compute(lhs, std::exp);
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = llvm::abs(lhs);
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);

    lhs.roundToIntegral(llvm::RoundingMode::TowardPositive);
    APFloat result = lhs;
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);

    lhs.roundToIntegral(llvm::RoundingMode::TowardNegative);
    APFloat result = lhs;
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = getFloatData(operands[2]);
    result.print(llvm::outs() << "result (before FMA): ");

    result = lhs * rhs + result;
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = compute(lhs, std::log10);
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = compute(lhs, std::log2);
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = compute(lhs, std::log);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMMinNumOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMMinNumOpInterpreter,
                                                   LLVM::MinNumOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::MinNum\n";
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = llvm::minnum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMMinimumOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMMinimumOpInterpreter,
                                                   LLVM::MinimumOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::Minimum\n";
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = llvm::minimum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMMaximumOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMMaximumOpInterpreter,
                                                   LLVM::MaximumOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::Maximum\n";
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = llvm::maximum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMMaxNumOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMMaxNumOpInterpreter,
                                                   LLVM::MaxNumOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::MaxNum\n";
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = llvm::maxnum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);
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
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = compute(lhs, std::sin);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSqrtOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSqrtOpInterpreter,
                                                   LLVM::SqrtOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::Sqrt\n";
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = compute(lhs, std::sqrt);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMPowOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMPowOpInterpreter,
                                                   LLVM::PowOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::Pow\n";
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = compute(lhs, rhs, std::pow);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMPowIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMPowIOpInterpreter,
                                                   LLVM::PowIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::PowI\n";
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APInt rhs = getIntegerData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", lhs);
    APFloat result = compute(lhs, rhs, std::pow);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSMaxOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSMaxOpInterpreter,
                                                   LLVM::SMaxOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting LLVM::SMax\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.sgt(rhs) ? lhs : rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);

    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMSMinOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMSMinOpInterpreter,
                                                   LLVM::SMinOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    llvm::outs() << "Interpreting LLVM::SMin\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.slt(rhs) ? lhs : rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMUMaxOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMUMaxOpInterpreter,
                                                   LLVM::UMaxOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting LLVM::UMax\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.ugt(rhs) ? lhs:rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMUMinOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMUMinOpInterpreter,
                                                   LLVM::UMinOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting LLVM::UMin\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.ult(rhs) ? lhs : rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMBitReverseOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMBitReverseOpInterpreter,
                                                   LLVM::BitReverseOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting LLVM::BitReverse\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt result = lhs.reverseBits();
    mli::fmt::printResult(llvm::outs(), result, isSigned);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMByteSwapOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMByteSwapOpInterpreter,
                                                   LLVM::ByteSwapOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting LLVM::ByteSwap\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt result = lhs.byteSwap();
    mli::fmt::printResult(llvm::outs(), result, isSigned);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMCopySignOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMCopySignOpInterpreter,
                                                   LLVM::CopySignOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::CopySign\n";
    APFloat result = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", result);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    result.copySign(rhs);
    mli::fmt::printResult(llvm::outs(), result);
        // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMCountLeadingZerosOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMCountLeadingZerosOpInterpreter,
                                                   LLVM::CountLeadingZerosOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting LLVM::CountLeadingZeros\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    unsigned result = lhs.countLeadingZeros();
    mli::fmt::printResult(llvm::outs(), result);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMCountTrailingZerosOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMCountTrailingZerosOpInterpreter,
                                                   LLVM::CountTrailingZerosOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting LLVM::CountLeadingZeros\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    unsigned result = lhs.countTrailingZeros();
    mli::fmt::printResult(llvm::outs(), result);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMCtPopOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMCtPopOpInterpreter,
                                                   LLVM::CtPopOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    llvm::outs() << "Interpreting LLVM::CtPop\n";
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);

    unsigned result = lhs.popcount();
    mli::fmt::printResult(llvm::outs(), result);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMAddressOfOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAddressOfOpInterpreter, LLVM::AddressOfOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    auto globalName = op->getAttrOfType<StringAttr>("global_name");
    llvm::outs() << mli::fmt::dim("global_name: ") << mli::fmt::highlight(globalName.getValue().str()) << "\n";

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &globalName, sizeof(globalName));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMAllocaOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMAllocaOpInterpreter, LLVM::AllocaOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    auto arraySize = operands[0].getData<int>().front();
    llvm::outs() << mli::fmt::dim("arraySize: ") << mli::fmt::highlight(std::to_string(arraySize)) << "\n";

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &arraySize, sizeof(arraySize));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct LLVMConstantOpInterpreter
    : public InterpreterOpInterface::ExternalModel<LLVMConstantOpInterpreter,
                                                   LLVM::ConstantOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    llvm::outs() << "Interpreting LLVM::Constant\n";
    auto att = op->getAttr("value");
    EvalValue evalResult;
    if (auto int_att = dyn_cast<IntegerAttr>(att)) {
        APInt result = int_att.getValue();
        result.print(llvm::outs() << "Initializing constant ", false);
        evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
            }
    else if (auto float_att = dyn_cast<FloatAttr>(att)) {
        APFloat result = float_att.getValue();
        result.print(llvm::outs() << "Initializing constant ");
        evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    }
    else {
        llvm::errs() << "Found constant that is neither integer nor float\n";
    }
    // Create an EvalValue from the result
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
    LLVM::ICmpOp::attachInterface<LLVMICmpOpInterpreter>(context);
    LLVM::FCmpOp::attachInterface<LLVMFCmpOpInterpreter>(context);

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
    LLVM::ConstantOp::attachInterface<LLVMConstantOpInterpreter>(context);

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
    LLVM::Exp2Op::attachInterface<LLVMExp2OpInterpreter>(context);
    LLVM::ExpOp::attachInterface<LLVMExpOpInterpreter>(context);
    LLVM::FAbsOp::attachInterface<LLVMFAbsOpInterpreter>(context);
    LLVM::FCeilOp::attachInterface<LLVMFCeilOpInterpreter>(context);
    LLVM::FFloorOp::attachInterface<LLVMFFloorOpInterpreter>(context);
    LLVM::FMAOp::attachInterface<LLVMFMAOpInterpreter>(context);
    LLVM::Log10Op::attachInterface<LLVMLog10OpInterpreter>(context);
    LLVM::Log2Op::attachInterface<LLVMLog2OpInterpreter>(context);
    LLVM::LogOp::attachInterface<LLVMLogOpInterpreter>(context);
    LLVM::MaximumOp::attachInterface<LLVMMaximumOpInterpreter>(context);
    LLVM::MinimumOp::attachInterface<LLVMMinimumOpInterpreter>(context);
    LLVM::MaxNumOp::attachInterface<LLVMMaxNumOpInterpreter>(context);
    LLVM::MinNumOp::attachInterface<LLVMMinNumOpInterpreter>(context);
    LLVM::PowOp::attachInterface<LLVMPowOpInterpreter>(context);
    LLVM::PowIOp::attachInterface<LLVMPowIOpInterpreter>(context);
    LLVM::SinOp::attachInterface<LLVMSinOpInterpreter>(context);
    LLVM::SMaxOp::attachInterface<LLVMSMaxOpInterpreter>(context);
    LLVM::SMinOp::attachInterface<LLVMSMinOpInterpreter>(context);
    LLVM::SqrtOp::attachInterface<LLVMSqrtOpInterpreter>(context);
    LLVM::UMaxOp::attachInterface<LLVMUMaxOpInterpreter>(context);
    LLVM::UMinOp::attachInterface<LLVMUMinOpInterpreter>(context);

    // Bit Manipulation Intrinsics
    LLVM::BitReverseOp::attachInterface<LLVMBitReverseOpInterpreter>(context);
    LLVM::ByteSwapOp::attachInterface<LLVMByteSwapOpInterpreter>(context);
    LLVM::CopySignOp::attachInterface<LLVMCopySignOpInterpreter>(context);
    LLVM::CountLeadingZerosOp::attachInterface<LLVMCountLeadingZerosOpInterpreter>(context);
    LLVM::CountTrailingZerosOp::attachInterface<LLVMCountTrailingZerosOpInterpreter>(context);
    LLVM::CtPopOp::attachInterface<LLVMCtPopOpInterpreter>(context);
    /*
    LLVM::FShlOp::attachInterface<LLVMFShlOpInterpreter>(context);
    LLVM::FShrOp::attachInterface<LLVMFShrOpInterpreter>(context);
    */
}
