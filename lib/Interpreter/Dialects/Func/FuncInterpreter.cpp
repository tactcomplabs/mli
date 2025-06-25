////===- FuncInterpreter.cpp - Func dialect interpreter -------------*- C++
///-*-===//
//
// Part of this file is part of the LLVM Project, under the Apache License v2.0
// with LLVM Exceptions. See https://llvm.org/LICENSE.txt for license
// information. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The remaining portions of this file are:
//
// Copyright (C) 2024-2025 Tactical Computing Laboratories, LLC
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
//
//===----------------------------------------------------------------------===//

#include "MLIFormat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interpreter/Dialects/FuncInterpreter.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"
#include "llvm/IR/Function.h"

using namespace mlir;

namespace {

class ReturnOpInterpreter : public InterpreterOpInterface::ExternalModel<ReturnOpInterpreter, func::ReturnOp> {
  public:
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        return interpreter.createReturnValueResult(operands);
    }
};

struct ConstantOpInterpreter : public InterpreterOpInterface::ExternalModel<ConstantOpInterpreter, func::ConstantOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
        auto func_attr = op->getAttr("value");

        // Get StringRef representing callee from attribute
        std::string              func_name;
        llvm::raw_string_ostream oss(func_name);
        func_attr.print(oss);
        if ( func_name[0] == '@' ) {
            func_name = func_name.substr(1);  // strip @ symbol from function name
        }

        auto func_op = interpreter.getModule().lookupSymbol<func::FuncOp>(func_name);
        if ( !func_op ) {
            return interpreter.createErrorResult("Can't create reference to unknown function " + func_name);
        }
        EvalValue func_ref = interpreter.createEvalValue(op->getResult(0).getType(), &func_op, sizeof(func_op));
        llvm::outs() << mli::fmt::dim("Creating reference to function " + func_name + "\n");
        return interpreter.createBindValueResult(func_ref);
    }
};

struct CallOpInterpreter : public InterpreterOpInterface::ExternalModel<CallOpInterpreter, func::CallOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
        auto callee_attr = op->getAttr("callee");
        // Get StringRef representing callee from attribute
        std::string              callee_name;
        llvm::raw_string_ostream oss(callee_name);
        callee_attr.print(oss);
        if ( callee_name[0] == '@' ) {
            callee_name = callee_name.substr(1);  // strip @ symbol from function name
        }

        llvm::outs() << mli::fmt::dim("Calling function: " + callee_name + "\n");
        return interpreter.execute(callee_name, operands);
    }
};

struct CallIndirectOpInterpreter : public InterpreterOpInterface::ExternalModel<CallIndirectOpInterpreter, func::CallIndirectOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        mli::fmt::printOpName(llvm::outs(), op->getName().getStringRef().str());
        auto function_ref = operands[0].getData<func::FuncOp>().front();
        llvm::outs() << mli::fmt::dim("Calling function " + function_ref.getName().str() + " from reference\n");
        // The first argument is the function itself, so slice it off to get the
        // function args
        return interpreter.execute(function_ref, operands.slice(1));
    }
};

}  // namespace

void FuncInterpreter::attachInterface(MLIRContext& context) {
    func::CallOp::attachInterface<CallOpInterpreter>(context);
    func::CallIndirectOp::attachInterface<CallIndirectOpInterpreter>(context);
    func::ConstantOp::attachInterface<ConstantOpInterpreter>(context);
    func::ReturnOp::attachInterface<ReturnOpInterpreter>(context);
}
