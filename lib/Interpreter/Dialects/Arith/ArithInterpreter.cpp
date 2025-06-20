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

using namespace mlir;
using namespace mli;

namespace {

#if NEW_LLVM
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
#endif

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

struct ArithAddUIExtendedOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithAddUIExtendedOpInterpreter,
                                                   arith::AddUIExtendedOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    bool overflow = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.uadd_ov(rhs, overflow);
    mli::fmt::printResult(llvm::outs(), std::make_pair(result, overflow), isSigned);
    // Create an EvalValue from the result
    EvalValue evalResults[2];
    evalResults[0] = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    evalResults[1] = interpreter.createEvalValue(op->getResult(1).getType(), &overflow, sizeof(overflow));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResults);
  }
};

struct ArithAndIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithAndIOpInterpreter,
                                                   arith::AndIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs & rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithBitcastOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithBitcastOpInterpreter,
                                                   arith::BitcastOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    auto src_type = operands[0].getType();
    auto result_type = op->getOpResult(0).getType();
    EvalValue evalResult;

    if (src_type.getIntOrFloatBitWidth() != result_type.getIntOrFloatBitWidth()) {
        return interpreter.createErrorResult("Types must be of the same width");
    }
    if (src_type == result_type) { // no-op
        evalResult = operands[0];
    }
    else if (llvm::isa<mlir::FloatType>(src_type)) { // src float, ret int
        APFloat lhs = getFloatData(operands[0]);
        mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
        APInt result = lhs.bitcastToAPInt();
        mli::fmt::printResult(llvm::outs(), result);
        evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
    }
    else if (llvm::isa<mlir::IntegerType>(src_type)) { // src int, ret float
        APInt lhs = getIntegerData(operands[0]);
        mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
        Semantics s = getFloatSemantics(result_type);
        APFloat result = APFloat(llvm::APFloat::EnumToSemantics(s), lhs);
        mli::fmt::printResult(llvm::outs(), result);
        evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
    }
    else {
        return interpreter.createErrorResult("Input is not numeric");
    }
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithCeilDivSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithCeilDivSIOpInterpreter,
                                                   arith::CeilDivSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);

    // APInt doesn't have a ceiling division function so rely on remainder
    APInt result, remainder;
    APInt::sdivrem(lhs, rhs, result, remainder);

    // If the quotient is negative, standard division behaves the same as ceiling
    // Otherwise, if the quotient isn't an integer, we need to add one for the ceiling
    if (remainder != 0 && result.isStrictlyPositive()) {
        result = ++result;
    }

    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithCeilDivUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithCeilDivUIOpInterpreter,
                                                   arith::CeilDivUIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);

    // APInt doesn't have a ceiling division function so rely on remainder
    APInt result, remainder;
    APInt::udivrem(lhs, rhs, result, remainder);

    if (remainder != 0) {
        result = ++result;
    }

    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
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
    bool one_NaN = lhs.isNaN() || rhs.isNaN();

    // FCmp supports both ordered and unordered comparisons
    // Any ordered comparison returns false if at least one argument is NaN
    // Any unordered comparison returns true if at least one argument is NaN
    switch (predicate) {
        case arith::CmpFPredicate::AlwaysFalse:
            result = false;
            break;
        case arith::CmpFPredicate::OEQ:
            result = !one_NaN && lhs == rhs;
            break;
        case arith::CmpFPredicate::OGT:
            result = !one_NaN && lhs > rhs;
            break;
        case arith::CmpFPredicate::OGE:
            result = !one_NaN && lhs >= rhs;
            break;
        case arith::CmpFPredicate::OLT:
            result = !one_NaN && lhs < rhs;
            break;
        case arith::CmpFPredicate::ONE:
            result = !one_NaN && lhs != rhs;
            break;
        case arith::CmpFPredicate::ORD:
            result = !one_NaN;
            break;
        case arith::CmpFPredicate::UEQ:
            result = one_NaN || lhs == rhs;
            break;
        case arith::CmpFPredicate::UGT:
            result = one_NaN || lhs > rhs;
            break;
        case arith::CmpFPredicate::UGE:
            result = one_NaN || lhs >= rhs;
            break;
        case arith::CmpFPredicate::ULT:
            result = one_NaN || lhs < rhs;
            break;
        case arith::CmpFPredicate::ULE:
            result = one_NaN || lhs <= rhs;
            break;
        case arith::CmpFPredicate::UNE:
            result = one_NaN || lhs != rhs;
            break;
        case arith::CmpFPredicate::UNO:
            result = one_NaN;
            break;
        default:
            result = true;
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
    auto result_type = op->getResult(0).getType();
    EvalValue evalResult;

    // Is integer or index type
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        APInt val = intAttr.getValue();
        if (mlir::isa<mlir::IndexType>(result_type)) {
            intptr_t result = static_cast<intptr_t>(val.sextOrTrunc(sizeof(intptr_t) * 8).getSExtValue());
            evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
            llvm::outs() << mli::fmt::dim("Initializing constant ") << result << "\n";
        }
        else {
            evalResult = interpreter.createEvalValue(result_type, &val, sizeof(val));
            llvm::outs() << mli::fmt::dim("Initializing constant ") << val << "\n";
        }
    } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
        APFloat result = floatAttr.getValue();
        evalResult = interpreter.createEvalValue(result_type, &result, sizeof(result));
        llvm::outs() << mli::fmt::dim("Initializing constant ") << result << "\n";
    } else {
        return interpreter.createErrorResult("Unsupported constant type");
    }
    return interpreter.createBindValueResult(evalResult);
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

struct ArithDivSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithDivSIOpInterpreter,
                                                   arith::DivSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.sdiv(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithDivUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithDivUIOpInterpreter,
                                                   arith::DivUIOp> {
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

struct ArithExtSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithExtSIOpInterpreter,
                                                   arith::ExtSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    unsigned new_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    APInt result = lhs.sext(new_width);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithExtUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithExtUIOpInterpreter,
                                                   arith::ExtUIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    unsigned new_width = op->getResult(0).getType().getIntOrFloatBitWidth();

    APInt result = lhs.zext(new_width);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithFloorDivSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithFloorDivSIOpInterpreter,
                                                   arith::FloorDivSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    
    bool overflow = false;
    APInt result = lhs.sfloordiv_ov(rhs, overflow);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
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

struct ArithIndexCastOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithIndexCastOpInterpreter,
                                                   arith::IndexCastOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    auto src_type = operands[0].getType();
    auto dest_type = op->getResult(0).getType();

    EvalValue evalResult;
    if (mlir::isa<mlir::IntegerType>(src_type) && mlir::isa<mlir::IndexType>(dest_type)) {
        APInt lhs = getIntegerData(operands[0], isSigned);
        mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
        intptr_t result = static_cast<intptr_t>(
            lhs.sextOrTrunc(sizeof(intptr_t) * 8).getSExtValue());
        mli::fmt::printResult(llvm::outs(), result, isSigned);
        evalResult = interpreter.createEvalValue(dest_type, &result, sizeof(result));
    }
    else if (mlir::isa<mlir::IndexType>(src_type) && mlir::isa<mlir::IntegerType>(dest_type)) {
        intptr_t lhs = operands[0].getData<intptr_t>().front();
        mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
        APInt result = APInt(dest_type.getIntOrFloatBitWidth(), lhs, isSigned);
        mli::fmt::printResult(llvm::outs(), result, isSigned);
        evalResult = interpreter.createEvalValue(dest_type, &result, sizeof(result));        
    }

    // Create an EvalValue from the result
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithIndexCastUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithIndexCastUIOpInterpreter,
                                                   arith::IndexCastUIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    auto src_type = operands[0].getType();
    auto dest_type = op->getResult(0).getType();

    EvalValue evalResult;
    if (mlir::isa<mlir::IntegerType>(src_type) && mlir::isa<mlir::IndexType>(dest_type)) {
        APInt lhs = getIntegerData(operands[0], isSigned);
        mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
        intptr_t result = static_cast<intptr_t>(
            lhs.sextOrTrunc(sizeof(intptr_t) * 8).getSExtValue());
        mli::fmt::printResult(llvm::outs(), result, isSigned);
        evalResult = interpreter.createEvalValue(dest_type, &result, sizeof(result));
    }
    else if (mlir::isa<mlir::IndexType>(src_type) && mlir::isa<mlir::IntegerType>(dest_type)) {
        intptr_t lhs = operands[0].getData<intptr_t>().front();
        mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
        APInt result = APInt(dest_type.getIntOrFloatBitWidth(), lhs, isSigned);
        mli::fmt::printResult(llvm::outs(), result, isSigned);
        evalResult = interpreter.createEvalValue(dest_type, &result, sizeof(result));        
    }
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
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

struct ArithMaxNumFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMaxNumFOpInterpreter,
                                                   arith::MaxNumFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    APFloat result = maxnum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMaxSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMaxSIOpInterpreter,
                                                   arith::MaxSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = APIntOps::smax(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMaxUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMaxUIOpInterpreter,
                                                   arith::MaxUIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = APIntOps::umax(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
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

struct ArithMinNumFOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMinNumFOpInterpreter,
                                                   arith::MinNumFOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APFloat lhs = getFloatData(operands[0]);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs);
    APFloat rhs = getFloatData(operands[1]);
    APFloat result = llvm::minnum(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMinSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMinSIOpInterpreter,
                                                   arith::MinSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = APIntOps::smin(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMinUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMinUIOpInterpreter,
                                                   arith::MinUIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = APIntOps::umin(lhs, rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
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

struct ArithMulIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMulIOpInterpreter,
                                                   arith::MulIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs * rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithMulSIExtendedOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMulSIExtendedOpInterpreter,
                                                   arith::MulSIExtendedOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);

    const unsigned op_width = operands[0].getType().getIntOrFloatBitWidth();
    lhs = lhs.sext(2 * op_width);
    lhs = lhs * rhs;
    APInt low = lhs.extractBits(op_width, 0);
    APInt high = lhs.extractBits(op_width, op_width);
    mli::fmt::printResult(llvm::outs(), std::make_pair(low, high), isSigned);

    // Create an EvalValue from the result
    EvalValue evalResults[2];
    evalResults[0] = interpreter.createEvalValue(op->getResult(0).getType(), &low, sizeof(low));
    evalResults[1] = interpreter.createEvalValue(op->getResult(1).getType(), &high, sizeof(high));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResults);
  }
};

struct ArithMulUIExtendedOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithMulUIExtendedOpInterpreter,
                                                   arith::MulUIExtendedOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);

    const unsigned op_width = operands[0].getType().getIntOrFloatBitWidth();
    lhs = lhs.sext(2 * op_width);
    lhs = lhs * rhs;
    APInt low = lhs.extractBits(op_width, 0);
    APInt high = lhs.extractBits(op_width, op_width);
    mli::fmt::printResult(llvm::outs(), std::make_pair(low, high), isSigned);

    // Create an EvalValue from the result
    EvalValue evalResults[2];
    evalResults[0] = interpreter.createEvalValue(op->getResult(0).getType(), &low, sizeof(low));
    evalResults[1] = interpreter.createEvalValue(op->getResult(1).getType(), &high, sizeof(high));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResults);
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

struct ArithOrIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithOrIOpInterpreter,
                                                   arith::OrIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    const APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    const APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs | rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
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

struct ArithRemSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithRemSIOpInterpreter,
                                                   arith::RemSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.srem(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithRemUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithRemUIOpInterpreter,
                                                   arith::RemUIOp> {
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

// Select Operation
struct ArithSelectOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithSelectOpInterpreter,
                                                   arith::SelectOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    bool condition = operands[0].getData<bool>().front();
    EvalValue result = condition ? operands[1] : operands[2];
    llvm::outs() << mli::fmt::dim("result: ") << result << "\n";
    return interpreter.createBindValueResult(result);
  }
};

struct ArithShLIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithShLIOpInterpreter,
                                                   arith::ShLIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], false);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, false);
    APInt result = lhs << rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithShRSIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithShRSIOpInterpreter,
                                                   arith::ShRSIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], false);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, false);
    APInt result = lhs.ashr(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithShRUIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithShRUIOpInterpreter,
                                                   arith::ShRUIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs.lshr(rhs);
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
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

struct ArithSubIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithSubIOpInterpreter,
                                                   arith::SubIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs-rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
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
    #if NEW_LLVM
    if (auto modeAttr =
        op->getAttrOfType<arith::RoundingModeAttr>("roundingmode")) {
        roundingMode = convertRoundingMode(modeAttr.getValue());
        llvm::outs() << mli::fmt::dim("Using specified rounding mode: ") << roundingMode << "\n";
    } 
    #endif
    llvm::outs() << mli::fmt::dim("Using default rounding mode: NearestTiesToEven") << "\n";

    bool losesInfo;
    auto resultType = op->getResult(0).getType();
    Semantics resultSemantics = getFloatSemantics(resultType);

    // Convert using the determined rounding mode
    APFloat result = operand;
    result.convert(llvm::APFloatBase::EnumToSemantics(resultSemantics),
                    roundingMode, &losesInfo);

    if (losesInfo) {
      llvm::outs() << mli::fmt::warning("Precision loss during truncation")
                   << "\n";
    }

    mli::fmt::printResult(llvm::outs(), result);

    auto evalResult =
        interpreter.createEvalValue(resultType, &result, sizeof(result));
    return interpreter.createBindValueResult(evalResult);
  }
};

struct ArithTruncIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithTruncIOpInterpreter,
                                                   arith::TruncIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = false;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt operand = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "operand", operand, isSigned);

    const unsigned result_width = op->getResult(0).getType().getIntOrFloatBitWidth();
    APInt result = operand.trunc(result_width);

    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
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

struct ArithXOrIOpInterpreter
    : public InterpreterOpInterface::ExternalModel<ArithXOrIOpInterpreter,
                                                   arith::XOrIOp> {
  static EvalResult interpret(Operation *op, Interpreter &interpreter,
                              ArrayRef<EvalValue> operands) {
    bool isSigned = true;
    mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
    APInt lhs = getIntegerData(operands[0], isSigned);
    mli::fmt::printOperand(llvm::outs(), "lhs", lhs, isSigned);
    APInt rhs = getIntegerData(operands[1], isSigned);
    mli::fmt::printOperand(llvm::outs(), "rhs", rhs, isSigned);
    APInt result = lhs ^ rhs;
    mli::fmt::printResult(llvm::outs(), result, isSigned);
    // Create an EvalValue from the result
    auto evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &result, sizeof(result));
    // Wrap the EvalValue in an ArrayRef and return the EvalResult
    return interpreter.createBindValueResult(evalResult);
  }
};

} // end anonymous namespace

void ArithInterpreter::attachInterface(MLIRContext &context) {
  arith::AddFOp::attachInterface<ArithAddFOpInterpreter>(context);
  arith::AddIOp::attachInterface<ArithAddIOpInterpreter>(context);
  arith::AddUIExtendedOp::attachInterface<ArithAddUIExtendedOpInterpreter>(context);
  arith::AndIOp::attachInterface<ArithAndIOpInterpreter>(context);
  arith::BitcastOp::attachInterface<ArithBitcastOpInterpreter>(context);
  arith::CeilDivSIOp::attachInterface<ArithCeilDivSIOpInterpreter>(context);
  arith::CeilDivUIOp::attachInterface<ArithCeilDivUIOpInterpreter>(context);
  arith::CmpFOp::attachInterface<ArithCmpFOpInterpreter>(context);
  arith::CmpIOp::attachInterface<ArithCmpIOpInterpreter>(context);
  arith::DivFOp::attachInterface<ArithDivFOpInterpreter>(context);
  arith::DivSIOp::attachInterface<ArithDivSIOpInterpreter>(context);
  arith::DivUIOp::attachInterface<ArithDivUIOpInterpreter>(context);
  arith::ExtFOp::attachInterface<ArithExtFOpInterpreter>(context);
  arith::ExtSIOp::attachInterface<ArithExtSIOpInterpreter>(context);
  arith::ExtUIOp::attachInterface<ArithExtUIOpInterpreter>(context);
  arith::ConstantOp::attachInterface<ArithConstantOpInterpreter>(context);
  arith::IndexCastOp::attachInterface<ArithIndexCastOpInterpreter>(context);
  arith::IndexCastUIOp::attachInterface<ArithIndexCastUIOpInterpreter>(context);
  arith::FloorDivSIOp::attachInterface<ArithFloorDivSIOpInterpreter>(context);
  arith::FPToSIOp::attachInterface<ArithFPToSIOpInterpreter>(context);
  arith::FPToUIOp::attachInterface<ArithFPToUIOpInterpreter>(context);
  arith::MaximumFOp::attachInterface<ArithMaximumFOpInterpreter>(context);
  arith::MaxNumFOp::attachInterface<ArithMaxNumFOpInterpreter>(context);
  arith::MaxSIOp::attachInterface<ArithMaxSIOpInterpreter>(context);
  arith::MaxUIOp::attachInterface<ArithMaxUIOpInterpreter>(context);
  arith::MinimumFOp::attachInterface<ArithMinimumFOpInterpreter>(context);
  arith::MinNumFOp::attachInterface<ArithMinNumFOpInterpreter>(context);
  arith::MinSIOp::attachInterface<ArithMinSIOpInterpreter>(context);
  arith::MinUIOp::attachInterface<ArithMinUIOpInterpreter>(context);
  arith::MulFOp::attachInterface<ArithMulFOpInterpreter>(context);
  arith::MulIOp::attachInterface<ArithMulIOpInterpreter>(context);
  arith::MulSIExtendedOp::attachInterface<ArithMulSIExtendedOpInterpreter>(context);
  arith::MulUIExtendedOp::attachInterface<ArithMulUIExtendedOpInterpreter>(context);
  arith::OrIOp::attachInterface<ArithOrIOpInterpreter>(context);
  arith::NegFOp::attachInterface<ArithNegFOpInterpreter>(context);
  arith::RemFOp::attachInterface<ArithRemFOpInterpreter>(context);
  arith::RemSIOp::attachInterface<ArithRemSIOpInterpreter>(context);
  arith::RemUIOp::attachInterface<ArithRemUIOpInterpreter>(context);
  arith::SelectOp::attachInterface<ArithSelectOpInterpreter>(context);
  arith::ShLIOp::attachInterface<ArithShLIOpInterpreter>(context);
  arith::ShRSIOp::attachInterface<ArithShRSIOpInterpreter>(context);
  arith::ShRUIOp::attachInterface<ArithShRUIOpInterpreter>(context);
  arith::SIToFPOp::attachInterface<ArithSIToFPOpInterpreter>(context);
  arith::SubFOp::attachInterface<ArithSubFOpInterpreter>(context);
  arith::TruncFOp::attachInterface<ArithTruncFOpInterpreter>(context);
  arith::TruncIOp::attachInterface<ArithTruncIOpInterpreter>(context);
  arith::UIToFPOp::attachInterface<ArithUIToFPOpInterpreter>(context);
  arith::XOrIOp::attachInterface<ArithXOrIOpInterpreter>(context);
}
