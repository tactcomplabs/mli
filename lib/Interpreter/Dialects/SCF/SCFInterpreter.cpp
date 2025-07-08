//===- SCFInterpreter.cpp - SCF dialect interpreter -------------*- C++ -*-===//
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

#include "mlir/Interpreter/Dialects/SCFInterpreter.h"
#include "MLIFormat.h"
#include "MLIUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"

using namespace mlir;
using namespace mli;

namespace {

// Build an EvalValue from an APInt
static EvalValue makeIntEvalValue(Interpreter& interp, Type ty, const APInt& v) {
  if ( ty.isIndex() ) {
    size_t word = v.getZExtValue();
    return interp.createEvalValue(ty, &word, sizeof(word));
  }
  const uint64_t* raw = v.getRawData();
  return interp.createEvalValue(ty, raw, v.getNumWords() * sizeof(uint64_t));
}


struct SCFYieldOpInterpreter : public InterpreterOpInterface::ExternalModel<SCFYieldOpInterpreter, scf::YieldOp> {
  static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
    fmt::printOpName(llvm::outs(), "scf.yield");

    llvm::outs() << fmt::dim("SCF Yield operands: ") << operands.size() << "\n";

    for ( size_t i = 0; i < operands.size(); ++i ){
      llvm::outs() << fmt::dim("  operand[" + std::to_string(i) + "] ") << operands[i].getRawDataSizeInBytes() << " B\n";
    }

    return interpreter.createYieldValueResult(operands);
  }
};


struct SCFForOpInterpreter : public InterpreterOpInterface::ExternalModel<SCFForOpInterpreter, scf::ForOp> {
  static EvalResult interpret(Operation* op, Interpreter& interp, ArrayRef<EvalValue> operands) {
    auto forOp = cast<scf::ForOp>(op);
    fmt::printOpName(llvm::outs(), "scf.for");

    if ( operands.size() < 3 ) {
      return interp.createErrorResult("scf.for expects lb, ub, step (+carried)");
    }

    APInt lb   = getIntegerData(operands[0]);
    APInt ub   = getIntegerData(operands[1]);
    APInt step = getIntegerData(operands[2]);

    if ( step == 0 ) {
      return interp.createErrorResult("scf.for step must be non-zero");
    }

    bool descending = step.isNegative();
    SmallVector<EvalValue, 4> carried(operands.begin() + 3, operands.end());

    for ( APInt iv = lb; descending ? iv.sgt(ub) : iv.slt(ub); iv += step ) {
      SmallVector<EvalValue, 4> args;
      args.push_back(makeIntEvalValue(interp, forOp.getInductionVar().getType(), iv));
      args.append(carried);

      EvalResult bodyRes = interp.execute(forOp.getRegion(), args);
      if ( bodyRes.getKind() != EvalResultKind::YieldValue ) {
          return interp.createErrorResult("scf.for body must yield");
      }

      if ( bodyRes.getValues().size() != forOp.getNumResults() ){
          return interp.createErrorResult("yield/result arity mismatch");
      }

      carried.assign(bodyRes.getValues().begin(), bodyRes.getValues().end());
    }

    fmt::printOpName(llvm::outs(), "scf.for end");
    return interp.createBindValueResult(carried);
  }
};


struct SCFIfOpInterpreter : public InterpreterOpInterface::ExternalModel<SCFIfOpInterpreter, scf::IfOp> {
  static EvalResult interpret(Operation* op, Interpreter& interp, ArrayRef<EvalValue> operands) {
    auto ifOp = cast<scf::IfOp>(op);
    fmt::printOpName(llvm::outs(), "scf.if");

    bool    cond   = getIntegerData(operands[0]).getBoolValue();
    Region& region = cond ? ifOp.getThenRegion() : ifOp.getElseRegion();

    EvalResult res = interp.execute(region, {});

    if ( res.getKind() != EvalResultKind::YieldValue ){
      return interp.createErrorResult("expected yield in scf.if region");
    }

    fmt::printOpName(llvm::outs(), "scf.if end");
    return interp.createBindValueResult(res.getValues());
  }
};


struct SCFWhileOpInterpreter : public InterpreterOpInterface::ExternalModel<SCFWhileOpInterpreter, scf::WhileOp> {
  static EvalResult interpret(Operation* op, Interpreter& interp, ArrayRef<EvalValue> operands) {
    auto whileOp = cast<scf::WhileOp>(op);
    fmt::printOpName(llvm::outs(), "scf.while");

    SmallVector<EvalValue, 4> carried(operands.begin(), operands.end());

    while ( true ) {
      EvalResult before = interp.execute(whileOp.getBefore(), carried);
      if ( before.getKind() != EvalResultKind::YieldValue )
        return interp.createErrorResult("expected scf.condition from before region");

      bool cond = getIntegerData(before.getValues()[0]).getBoolValue();
      if ( !cond ) {
        SmallVector<EvalValue, 4> results(before.getValues().begin() + 1, before.getValues().end());
        return interp.createBindValueResult(results);
      }

      SmallVector<EvalValue, 4> afterArgs(before.getValues().begin() + 1, before.getValues().end());
      EvalResult                after = interp.execute(whileOp.getAfter(), afterArgs);
      if ( after.getKind() != EvalResultKind::YieldValue ){
        return interp.createErrorResult("expected yield from after region");
      }

      carried.assign(after.getValues().begin(), after.getValues().end());
  }
}
};


struct SCFConditionOpInterpreter : public InterpreterOpInterface::ExternalModel<SCFConditionOpInterpreter, scf::ConditionOp> {
  static EvalResult interpret(Operation* op, Interpreter& interp, ArrayRef<EvalValue> operands) {
    fmt::printOpName(llvm::outs(), "scf.condition");

    if ( operands.empty() ) {
      return interp.createErrorResult("scf.condition requires at least a condition operand");
    }

    return interp.createYieldValueResult(operands);
  }
};

}  // end anonymous namespace

void SCFInterpreter::attachInterface(MLIRContext& ctx) {
    scf::YieldOp::attachInterface<SCFYieldOpInterpreter>(ctx);
    scf::ForOp::attachInterface<SCFForOpInterpreter>(ctx);
    scf::IfOp::attachInterface<SCFIfOpInterpreter>(ctx);
    scf::WhileOp::attachInterface<SCFWhileOpInterpreter>(ctx);
    scf::ConditionOp::attachInterface<SCFConditionOpInterpreter>(ctx);
}
