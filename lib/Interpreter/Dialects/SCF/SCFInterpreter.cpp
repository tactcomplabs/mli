//===- SCFInterpreter.cpp - SCF dialect interpreter -------------*- C++ -*-===//
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

#include "MLIFormat.h"
#include "MLIUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interpreter/Dialects/SCFInterpreter.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"

using namespace mlir;
using namespace mli;

namespace {

// Build an EvalValue from an APInt
static EvalValue makeIntEvalValue(Interpreter& interp, Type ty, const APInt& v) {
    if ( ty.isIndex() ) {
        intptr_t word = v.getZExtValue();
        return interp.createEvalValue(ty, &word, sizeof(word));
    }
    // Not index, just use APInt
    return interp.createEvalValue(ty, &v, sizeof(v));
}

struct SCFYieldOpInterpreter : public InterpreterOpInterface::ExternalModel<SCFYieldOpInterpreter, scf::YieldOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        fmt::printOpName(llvm::outs(), "scf.yield");

        llvm::outs() << fmt::dim("SCF Yield operands: ") << operands.size() << "\n";

        for ( size_t i = 0; i < operands.size(); ++i ) {
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

        // The bounds and steps can be either index or integer types
        // We'll use integers, but must get them as intptr_t if they're listed as indices to avoid UB in casting
        constexpr int LOOP_ARGS = 3;
        APInt         loopVars[LOOP_ARGS];
        for ( size_t i = 0; i < LOOP_ARGS; i++ ) {
            Type t = operands[i].getType();
            if ( t.isIndex() ) {
                intptr_t val_as_idx = operands[i].getData<intptr_t>().front();
                loopVars[i]         = APInt(8 * sizeof(intptr_t), val_as_idx);
            }
            else {
                loopVars[i] = operands[i].getIntegerData();
            }
        }

        APInt lb   = loopVars[0];
        APInt ub   = loopVars[1];
        APInt step = loopVars[2];

        // Required by MLIR spec (see https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)
        if ( step.isNonPositive() ) {
            return interp.createErrorResult("scf.for step must be positive");
        }

        // Arguments to be carried through each iteration, if any
        SmallVector<EvalValue, 4> carried(operands.begin() + 3, operands.end());
        APInt                     iv = lb;
        for ( ; iv.slt(ub); iv += step ) {
            SmallVector<EvalValue, 4> args;
            args.push_back(makeIntEvalValue(interp, forOp.getInductionVar().getType(), iv));
            args.append(carried);

            EvalResult bodyRes = interp.execute(forOp.getRegion(), args);
            // Propagate any error message from the for body
            if ( bodyRes.getKind() == EvalResultKind::Error ) {
                return bodyRes;
            }
            if ( bodyRes.getKind() != EvalResultKind::YieldValue ) {
                return interp.createErrorResult("scf.for body must yield");
            }

            if ( bodyRes.getValues().size() != forOp.getNumResults() ) {
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

        bool    cond   = operands[0].getData<bool>().front();
        Region& region = cond ? ifOp.getThenRegion() : ifOp.getElseRegion();

        // When scf.if doesn't produce results, the else block is optional
        // If the else block is omitted, the first block in ifOp.getElseRegion() has 424967154 args for some reason
        // Fall through, skipping the call to execute a region
        if ( !cond && op->getNumResults() == 0 ) {
            return interp.createVoidResult();
        }

        EvalResult res = interp.execute(region, {});
        fmt::printOpName(llvm::outs(), "scf.if end");

        // Propagate any errors from the previous execute call
        if ( res.getKind() == EvalResultKind::Error ) {
            return res;
        }
        // No results produced => no SSA value to bind
        if ( op->getNumResults() == 0 ) {
            return interp.createVoidResult();
        }
        // We must have a yield statement in both blocks if we are producing results
        if ( res.getKind() != EvalResultKind::YieldValue ) {
            return interp.createErrorResult("expected yield in scf.if region");
        }
        return interp.createBindValueResult(res.getValues());
    }
};

struct SCFIndexSwitchOpInterpreter : public InterpreterOpInterface::ExternalModel<SCFIndexSwitchOpInterpreter, scf::IndexSwitchOp> {
    static EvalResult interpret(Operation* op, Interpreter& interp, ArrayRef<EvalValue> operands) {
        scf::IndexSwitchOp switchOp = cast<scf::IndexSwitchOp>(op);
        fmt::printOpName(llvm::outs(), "scf.index_switch");

        const intptr_t          target = operands[0].getData<intptr_t>().front();
        llvm::ArrayRef<int64_t> cases  = switchOp.getCases();

        EvalResult region_result;
        bool       foundRegion = false;
        for ( int i = 0; i < cases.size(); i++ ) {
            if ( cases[i] == target ) {
                llvm::outs() << mli::fmt::dim("Taking case " + std::to_string(cases[i]) + "\n");
                mlir::Region& target_region = switchOp.getCaseRegions()[i];
                region_result               = interp.execute(target_region, operands.slice(1));
                foundRegion                 = true;
                break;
            }
        }

        // No matches found, use default case
        if ( !foundRegion ) {
            llvm::outs() << mli::fmt::dim("Taking default case\n");
            mlir::Region& target_region = switchOp.getDefaultRegion();
            region_result               = interp.execute(target_region, operands.slice(1));
        }

        // Propagate errors from case block
        if ( region_result.getKind() == EvalResultKind::Error ) {
            return region_result;
        }
        if ( op->getNumResults() == 0 ) {
            return interp.createVoidResult();
        }
        // Ensure case yields a value if operation is expecting a result
        if ( region_result.getKind() != EvalResultKind::YieldValue ) {
            return interp.createErrorResult("scf.index_switch case must yield if expecting results");
        }
        return interp.createBindValueResult(region_result.getValues());
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

            bool cond = before.getValues()[0].getData<bool>().front();
            // Condition is false, terminate while loop
            if ( !cond ) {
                SmallVector<EvalValue, 4> results(before.getValues().begin() + 1, before.getValues().end());
                return interp.createBindValueResult(results);
            }

            SmallVector<EvalValue, 4> afterArgs(before.getValues().begin() + 1, before.getValues().end());
            EvalResult                after = interp.execute(whileOp.getAfter(), afterArgs);
            if ( after.getKind() != EvalResultKind::YieldValue ) {
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
    scf::IndexSwitchOp::attachInterface<SCFIndexSwitchOpInterpreter>(ctx);
    scf::WhileOp::attachInterface<SCFWhileOpInterpreter>(ctx);
    scf::ConditionOp::attachInterface<SCFConditionOpInterpreter>(ctx);
}
