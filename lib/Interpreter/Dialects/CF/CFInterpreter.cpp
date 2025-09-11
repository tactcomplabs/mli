//===- CFInterpreter.cpp - CF dialect interpreter -------------*- C++ -*-===//
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
#include "mlir/Interpreter/Dialects/CFInterpreter.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"

using namespace mlir;
using namespace mli;

namespace {
struct AssertOpInterpreter : public InterpreterOpInterface::ExternalModel<AssertOpInterpreter, cf::AssertOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        bool cond = operands[0].getData<bool>().front();
        if ( !cond ) {
            auto msg_attr = op->getAttrOfType<mlir::StringAttr>("msg");
            return interpreter.createErrorResult(msg_attr.str());
        }
        // Assert is a no-op if the flag is true
        // Create a "void" result, which doesn't branch, yield, error, or bind
        // It contains no useful information, but is necessary for passing around in execute()
        return interpreter.createVoidResult();
    }
};

struct BranchOpInterpreter : public InterpreterOpInterface::ExternalModel<BranchOpInterpreter, cf::BranchOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        if ( !op->hasSuccessors() ) {
            std::string              msg = "Block ";
            llvm::raw_string_ostream os(msg);
            op->getBlock()->printAsOperand(os);
            os << " does not have a successor";
            return interpreter.createErrorResult(os.str());
        }
        Block* succ = op->getSuccessor(0);
        return interpreter.createBranchResult(*succ, operands);
    }
};

struct CondBranchOpInterpreter : public InterpreterOpInterface::ExternalModel<CondBranchOpInterpreter, cf::CondBranchOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        bool cond = operands[0].getData<bool>().front();
        if ( !op->hasSuccessors() ) {
            std::string              msg = "Block ";
            llvm::raw_string_ostream os(msg);
            op->getBlock()->printAsOperand(os);
            os << " does not have a successor";
            return interpreter.createErrorResult(os.str());
        }
        Block* succ              = op->getSuccessor(!cond);  // take first successor if cond is true

        // operands has the structure [<cond>, <block1_args>, <block2_args>]
        // Select the subarray corresponding to the taken block's arguments
        const unsigned num_args  = succ->getNumArguments();
        unsigned       start_idx = cond ? 1 : operands.size() - num_args;
        return interpreter.createBranchResult(*succ, operands.slice(start_idx, num_args));
    }
};

struct SwitchOpInterpreter : public InterpreterOpInterface::ExternalModel<SwitchOpInterpreter, cf::SwitchOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        auto        case_val_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>("case_values");
        const APInt target        = operands[0].getIntegerData();

        // Iterator for successor blocks to jump to
        // Skip the first block and its argument list, since that's the default block
        auto     block_it         = op->successor_begin();
        unsigned start_idx        = 1 + (*block_it)->getNumArguments();
        ++block_it;  // skip default block for now

        for ( const APInt& case_val : case_val_attr ) {
            unsigned current_block_args = (*block_it)->getNumArguments();
            if ( case_val == target ) {
                llvm::outs() << mli::fmt::dim("Taking case: " + mli::fmt::to_string(case_val) + "\n");
                return interpreter.createBranchResult(**block_it, operands.slice(start_idx, current_block_args));
            }
            start_idx += current_block_args;
            ++block_it;  // NOTE: I'm assuming that numSuccessors == numCaseVals, otherwise this may go OOB
        }

        // Doesn't match any of the cases, use default
        llvm::outs() << mli::fmt::dim("Taking default case\n");
        Block* default_block = op->getSuccessor(0);
        return interpreter.createBranchResult(*default_block, operands.slice(1, default_block->getNumArguments()));
    }
};

}  // end anonymous namespace

void CFInterpreter::attachInterface(MLIRContext& context) {
    cf::AssertOp::attachInterface<AssertOpInterpreter>(context);
    cf::BranchOp::attachInterface<BranchOpInterpreter>(context);
    cf::CondBranchOp::attachInterface<CondBranchOpInterpreter>(context);
    cf::SwitchOp::attachInterface<SwitchOpInterpreter>(context);
}
