//===- ArithInterpreter.cpp - Arith dialect interpreter -------------*- C++
//-*-===//
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
#include "mlir/Interpreter/Dialects/ArithInterpreter.h"
#include "MLIFormat.h"
#include "MLIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace mli;
typedef llvm::APFloatBase::Semantics Semantics;

namespace {

static llvm::RoundingMode convertRoundingMode(arith::RoundingMode mode) {
  switch (mode) {
  case arith::RoundingMode::upward:
    return llvm::RoundingMode::TowardPositive;
  case arith::RoundingMode::downward:
    return llvm::RoundingMode::TowardNegative;
  case arith::RoundingMode::toward_zero:
    return llvm::RoundingMode::TowardZero;
  case arith::RoundingMode::to_nearest_even:
    return llvm::RoundingMode::NearestTiesToEven;
  case arith::RoundingMode::to_nearest_away:
    return llvm::RoundingMode::NearestTiesToAway;
  default:
    return llvm::RoundingMode::NearestTiesToEven;
  }
}

// Addition Operations
struct ArithAddFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithAddFOpInterpreter,
                                                   arith::AddFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);
    APFloat result = lhs + rhs;
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithAddIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithAddIOpInterpreter,
                                                   arith::AddIOp> {
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

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

// Comparison Operations
struct ArithCmpFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithCmpFOpInterpreter,
                                                   arith::CmpFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    auto predicate =
        op->getAttrOfType<arith::CmpFPredicateAttr>("predicate").getValue();

    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    bool result;
    switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      result = lhs == rhs;
      break;
    case arith::CmpFPredicate::OGT:
      result = lhs > rhs;
      break;
    case arith::CmpFPredicate::OGE:
      result = lhs >= rhs;
      break;
    case arith::CmpFPredicate::OLT:
      result = lhs < rhs;
      break;
    case arith::CmpFPredicate::OLE:
      result = lhs <= rhs;
      break;
    case arith::CmpFPredicate::ONE:
      result = lhs != rhs;
      break;
    default:
      result = false;
    }

    mli::fmt::printResult(llvm::outs(), result);
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithCmpIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithCmpIOpInterpreter,
                                                   arith::CmpIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    auto predicate =
        op->getAttrOfType<arith::CmpIPredicateAttr>("predicate").getValue();
    bool isSigned = predicate == arith::CmpIPredicate::slt ||
                    predicate == arith::CmpIPredicate::sle ||
                    predicate == arith::CmpIPredicate::sgt ||
                    predicate == arith::CmpIPredicate::sge;

    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);

    bool result;
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      result = lhs.eq(rhs);
      break;
    case arith::CmpIPredicate::ne:
      result = lhs.ne(rhs);
      break;
    case arith::CmpIPredicate::slt:
      result = lhs.slt(rhs);
      break;
    case arith::CmpIPredicate::sle:
      result = lhs.sle(rhs);
      break;
    case arith::CmpIPredicate::sgt:
      result = lhs.sgt(rhs);
      break;
    case arith::CmpIPredicate::sge:
      result = lhs.sge(rhs);
      break;
    case arith::CmpIPredicate::ult:
      result = lhs.ult(rhs);
      break;
    case arith::CmpIPredicate::ule:
      result = lhs.ule(rhs);
      break;
    case arith::CmpIPredicate::ugt:
      result = lhs.ugt(rhs);
      break;
    case arith::CmpIPredicate::uge:
      result = lhs.uge(rhs);
      break;
    default:
      result = false;
    }

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    mli::fmt::printResult(llvm::outs(), result);
    return interpreter.createBindValueResult(evalResult);
  }
};

// Constant Operation
struct ArithConstantOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithConstantOpInterpreter,
                                                   arith::ConstantOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    auto attr = op->getAttr("value");
    EvalValue evalResult;

    // NOTE: that variables with type index will be successfully casted here.
    // The only notable difference between integers and index is that the
    // latter's width is machine-dependent. MLIR doesn't define an IndexAttr, so
    // I don't think this will be a big issue.

    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      APInt result = intAttr.getValue();
      result.print(llvm::outs() << "Initializing constant ", false);
      evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                               &result, sizeof(result));
    } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
      APFloat result = floatAttr.getValue();
      result.print(llvm::outs() << "Initializing constant ");
      evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                               &result, sizeof(result));
    } else {
      llvm::errs() << "Unsupported constant type\n";
    }

    return interpreter.createBindValueResult(evalResult);
  }
};

// Maximum/Minimum Operations
struct ArithMaximumFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMaximumFOpInterpreter,
                                                   arith::MaximumFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    APFloat result = llvm::maximum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMinimumFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMinimumFOpInterpreter,
                                                   arith::MinimumFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    APFloat result = llvm::minimum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

// Select Operation
struct ArithSelectOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithSelectOpInterpreter,
                                                   arith::SelectOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool condition = operands[0].getData<bool>().front();

    EvalValue result = condition ? operands[1] : operands[2];
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    return interpreter.createBindValueResult(result);
  }
};

// Floating point division
struct ArithDivFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithDivFOpInterpreter,
                                                   arith::DivFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    APFloat result = lhs / rhs;
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithExtFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithExtFOpInterpreter,
                                                   arith::ExtFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat operand = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "operand", operand);

    bool losesInfo;
    auto resultType = op->getResult(0).getType();
    Semantics resultSemantics = getFloatSemantics(resultType);
    operand.convert(llvm::APFloatBase::EnumToSemantics(resultSemantics),
                    llvm::RoundingMode::TowardZero, &losesInfo);

    APFloat result = operand;
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult =
        interpreter.createEvalValue(resultType, &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMaxNumFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMaxNumFOpInterpreter,
                                                   arith::MaxNumFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    APFloat result = llvm::maxnum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMinNumFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMinNumFOpInterpreter,
                                                   arith::MinNumFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    APFloat result = llvm::minnum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMulFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMulFOpInterpreter,
                                                   arith::MulFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    APFloat result = lhs * rhs;
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithNegFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithNegFOpInterpreter,
                                                   arith::NegFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat operand = getFloatData(operands[0]);
    // mli::fmt::printOperand(llvm::outs(), "operand", operand);

    APFloat result = -operand;
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};
struct ArithFNegOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithFNegOpInterpreter,
                                                   arith::NegFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat result = -lhs;
    mli::fmt::printResult(llvm::outs(), result);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithRemFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithRemFOpInterpreter,
                                                   arith::RemFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    lhs.mod(rhs); // Modifies lhs in-place
    APFloat result = lhs;
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithSubFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithSubFOpInterpreter,
                                                   arith::SubFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs);

    APFloat result = lhs - rhs;
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithTruncFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithTruncFOpInterpreter,
                                                   arith::TruncFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat operand = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "operand", operand);

    // Get rounding mode from optional attribute
    llvm::RoundingMode roundingMode = llvm::RoundingMode::NearestTiesToEven;
    if (auto modeAttr =
        op->getAttrOfType<arith::RoundingModeAttr>("roundingmode")) {
        roundingMode = convertRoundingMode(modeAttr.getValue());
        llvm::outs() << mli::fmt::dim("Using specified rounding mode: ") << roundingMode << "\n";
    } else {
        llvm::outs() << mli::fmt::dim("Using default rounding mode: NearestTiesToEven") << "\n";
    }

    bool losesInfo;
    auto resultType = op->getResult(0).getType();
    Semantics resultSemantics = getFloatSemantics(resultType);

    // Convert using the determined rounding mode
    operand.convert(llvm::APFloatBase::EnumToSemantics(resultSemantics),
                    roundingMode, &losesInfo);

    if (losesInfo) {
      llvm::outs() << mli::fmt::warning("Precision loss during truncation")
                   << "\n";
    }

    APFloat result = operand;
    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult =
        interpreter.createEvalValue(resultType, &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithSIToFPOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithSIToFPOpInterpreter,
                                                   arith::SIToFPOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    bool isSigned = true;
    APInt operand = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "operand", operand, isSigned);

    auto resultType = op->getResult(0).getType();
    Semantics resultSemantics = getFloatSemantics(resultType);
    APFloat result(llvm::APFloatBase::EnumToSemantics(resultSemantics));
    result.convertFromAPInt(operand, isSigned, llvm::RoundingMode::TowardZero);

    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult =
        interpreter.createEvalValue(resultType, &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithUIToFPOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithUIToFPOpInterpreter,
                                                   arith::UIToFPOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    bool isSigned = false;
    APInt operand = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "operand", operand, isSigned);

    auto resultType = op->getResult(0).getType();
    Semantics resultSemantics = getFloatSemantics(resultType);
    APFloat result(llvm::APFloatBase::EnumToSemantics(resultSemantics));
    result.convertFromAPInt(operand, isSigned, llvm::RoundingMode::TowardZero);

    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult =
        interpreter.createEvalValue(resultType, &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithFPToSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithFPToSIOpInterpreter,
                                                   arith::FPToSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat operand = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "operand", operand);

    unsigned resultWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
    bool isUnsigned = false;
    APSInt result(resultWidth, isUnsigned);
    bool isExact;
    operand.convertToInteger(result, llvm::RoundingMode::TowardZero, &isExact);

    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithFPToUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithFPToUIOpInterpreter,
                                                   arith::FPToUIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat operand = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "operand", operand);

    unsigned resultWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
    bool isUnsigned = true;
    APSInt result(resultWidth, isUnsigned);
    bool isExact;
    operand.convertToInteger(result, llvm::RoundingMode::TowardZero, &isExact);

    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(),
                                                  &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

} // end anonymous namespace

void ArithInterpreter::attachInterface(MLIRContext &context) {
  arith::AddFOp::attachInterface<ArithAddFOpInterpreter>(context);
  arith::AddIOp::attachInterface<ArithAddIOpInterpreter>(context);
  arith::CmpFOp::attachInterface<ArithCmpFOpInterpreter>(context);
  arith::CmpIOp::attachInterface<ArithCmpIOpInterpreter>(context);
  arith::ConstantOp::attachInterface<ArithConstantOpInterpreter>(context);
  arith::MaximumFOp::attachInterface<ArithMaximumFOpInterpreter>(context);
  arith::MinimumFOp::attachInterface<ArithMinimumFOpInterpreter>(context);
  arith::SelectOp::attachInterface<ArithSelectOpInterpreter>(context);
  arith::SelectOp::attachInterface<ArithSelectOpInterpreter>(context);
  arith::DivFOp::attachInterface<ArithDivFOpInterpreter>(context);
  arith::ExtFOp::attachInterface<ArithExtFOpInterpreter>(context);
  arith::MaxNumFOp::attachInterface<ArithMaxNumFOpInterpreter>(context);
  arith::MinNumFOp::attachInterface<ArithMinNumFOpInterpreter>(context);
  arith::MulFOp::attachInterface<ArithMulFOpInterpreter>(context);
  arith::NegFOp::attachInterface<ArithNegFOpInterpreter>(context);
  arith::RemFOp::attachInterface<ArithRemFOpInterpreter>(context);
  arith::SubFOp::attachInterface<ArithSubFOpInterpreter>(context);
  arith::TruncFOp::attachInterface<ArithTruncFOpInterpreter>(context);
  arith::SIToFPOp::attachInterface<ArithSIToFPOpInterpreter>(context);
  arith::UIToFPOp::attachInterface<ArithUIToFPOpInterpreter>(context);
  arith::FPToSIOp::attachInterface<ArithFPToSIOpInterpreter>(context);
  arith::FPToUIOp::attachInterface<ArithFPToUIOpInterpreter>(context);
}
