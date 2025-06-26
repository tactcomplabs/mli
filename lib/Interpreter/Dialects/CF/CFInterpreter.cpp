//===- CFInterpreter.cpp - CF dialect interpreter -------------*- C++ -*-===//
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

#include "mlir/Interpreter/Dialects/CFInterpreter.h"
#include "MLIFormat.h"
#include "MLIUtils.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"

using namespace mlir;
using namespace mli;

namespace {
    struct AssertOpInterpreter
        : public InterpreterOpInterface::ExternalModel<AssertOpInterpreter,
                                                       cf::AssertOp> {
      static EvalResult interpret(Operation *op, Interpreter &interpreter,
                                  ArrayRef<EvalValue> operands) {
        bool cond = operands[0].getData<bool>().front();
        if (!cond) {
            auto msg_attr = op->getAttrOfType<mlir::StringAttr>("msg");
            return interpreter.createErrorResult(msg_attr.str());
        }
        // Assert is a no-op if the flag is true
        // Create a "void" result, which doesn't branch, yield, error, or bind
        // It contains no useful information, but is necessary for passing around in execute()
        return interpreter.createVoidResult();
      }
    };

    struct BranchOpInterpreter
        : public InterpreterOpInterface::ExternalModel<BranchOpInterpreter,
                                                       cf::BranchOp> {
      static EvalResult interpret(Operation *op, Interpreter &interpreter,
                                  ArrayRef<EvalValue> operands) {
        Block* succ = op->getSuccessor(0);
        return interpreter.createBranchResult(*succ, operands);
      }
    };

    struct CondBranchOpInterpreter
        : public InterpreterOpInterface::ExternalModel<CondBranchOpInterpreter,
                                                       cf::CondBranchOp> {
      static EvalResult interpret(Operation *op, Interpreter &interpreter,
                                  ArrayRef<EvalValue> operands) {
        bool cond = operands[0].getData<bool>().front();
        Block* succ = op->getSuccessor(!cond); // take first successor if cond is true
        return interpreter.createBranchResult(*succ, operands);
      }
    };

} // end anonymous namespace

void CFInterpreter::attachInterface(MLIRContext &context) {
    cf::AssertOp::attachInterface<AssertOpInterpreter>(context);
    cf::BranchOp::attachInterface<BranchOpInterpreter>(context);
    cf::CondBranchOp::attachInterface<CondBranchOpInterpreter>(context);
}
