////===- Interpreter.cpp - Interpreter Core -------------*- C++ -*-===//
//
// Part of this file is part of the LLVM Project, under the Apache License v2.0
// with LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The remaining portions of this file are:
//
// Copyright (C) 2024 Tactical Computing Laboratories, LLC
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
#include "mlir/Interpreter/Interpreter.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

EvalValue::EvalValue() : impl(nullptr) {}
EvalValue::~EvalValue() = default;
EvalValue::EvalValue(const EvalValue &src) = default;
EvalValue &EvalValue::operator=(const EvalValue &src) = default;
EvalValue::EvalValue(EvalValue &&src) = default;
EvalValue &EvalValue::operator=(EvalValue &&src) = default;
EvalValue::EvalValue(detail::EvalValueImpl *impl) : impl(impl) {}

Type EvalValue::getType() const { return impl->getType(); }

size_t EvalValue::getRawDataSizeInBytes() const {
  return impl->getRawDataSizeInBytes();
}

char *EvalValue::getRawData() { return impl->getRawData(); }

const char *EvalValue::getRawData() const { return impl->getRawData(); }

Interpreter::Interpreter(MLIRContext &context, bool enableStackTraceOnError)
    : context(&context), enableStackTraceOnError(enableStackTraceOnError) {}

EvalResult Interpreter::execute(StringRef entryFuncName,
                                ArrayRef<EvalValue> arguments) {
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(entryFuncName);
  if (!func) {
    return createErrorResult("failed to find function by name");
  }
  return execute(func, arguments);
}

EvalResult Interpreter::execute(StringRef entryFuncName,
                                ArgumentProviderRef argumentProvider) {
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(entryFuncName);
  if (!func) {
    return createErrorResult("failed to find function by name");
  }
  return execute(func, argumentProvider);
}

EvalResult Interpreter::execute(func::FuncOp func,
                                ArrayRef<EvalValue> arguments) {
  ScopedFunctionFrame functionFrameGuard(*this);
  return execute(func.getBody(), arguments);
}

EvalResult Interpreter::execute(func::FuncOp func,
                                ArgumentProviderRef argumentProvider) {
  EvalResult arguments =
      argumentProvider(*this, func.getFunctionType().getInputs());
  if (arguments.getKind() != EvalResultKind::BindValue) {
    return arguments;
  }
  return execute(func, arguments.getValues());
}

EvalResult Interpreter::execute(Region &region, ArrayRef<EvalValue> arguments) {
  // Check the region kind.
  if (Operation *parent_op = region.getParentOp()) {
    if (parent_op->isRegistered()) {
      if (auto interface = dyn_cast<RegionKindInterface>(parent_op)) {
        if (interface.getRegionKind(region.getRegionNumber()) !=
            RegionKind::SSACFG) {
          return createErrorResult("only SSACFG region can be interpreted");
        }
      }
    }
  }

  ScopedRegionFrame regionFrameGuard(*this);

  // Executes the operations in the blocks.
  EvalResult result = createBranchResult(region.front(), arguments);
  while (true) {
    result = execute(*result.getBlock(), arguments);
    switch (result.getKind()) {
    case EvalResultKind::Error:
      return result;
    case EvalResultKind::BindValue:
      return createErrorResult(
          "region evaluation result must be return or yield");
    case EvalResultKind::ReturnValue:
    case EvalResultKind::YieldValue:
      return result;
    case EvalResultKind::Branch:
      break;
    }
  }
}

EvalResult Interpreter::execute(Block &block, ArrayRef<EvalValue> arguments) {
  if (failed(setEvalValues(ValueRange(block.getArguments()), arguments))) {
    return createErrorResult(
        "number of formal arguments and actual arguments mismatch");
  }

  for (auto &op : block) {
    EvalResult result = execute(op);
    switch (result.getKind()) {
    case EvalResultKind::Error:
      return result;
    case EvalResultKind::BindValue: {
      if (failed(
              setEvalValues(ValueRange(op.getResults()), result.getValues()))) {
        return createErrorResult("number of op results mismatch");
      }
      continue;
    }
    case EvalResultKind::ReturnValue:
    case EvalResultKind::YieldValue:
    case EvalResultKind::Branch:
      return result;
    }
  }

  return createErrorResult("block does not end with a terminator op");
}

EvalResult Interpreter::execute(Operation &op) {
  llvm::SmallVector<EvalValue, 8> operands;
  for (Value operand : op.getOperands()) {
    operands.push_back(getEvalValue(operand));
  }
  return execute(op, operands);
}

EvalResult Interpreter::execute(Operation &op, ArrayRef<EvalValue> operands) {
  if (auto interface = dyn_cast<InterpreterOpInterface>(&op)) {
    return interface.interpret(*this, operands);
  }

  return createErrorResult("op does not implement InterpreterOpInterface: " +
                           debugString(op));
}

LogicalResult Interpreter::setEvalValues(ValueRange ssaNames,
                                         ArrayRef<EvalValue> evalValues) {
  if (ssaNames.size() != evalValues.size()) {
    return LogicalResult::failure();
  }
  for (auto pair : llvm::zip(ssaNames, evalValues)) {
    setEvalValue(std::get<0>(pair), std::get<1>(pair));
  }
  return LogicalResult::success();
}

EvalResult Interpreter::createErrorResult(StringRef errorMessage) {
  return EvalResult(
      EvalResultKind::Error, {}, nullptr,
      std::make_unique<EvalError>(errorMessage, enableStackTraceOnError));
}

EvalValue Interpreter::createEvalValue(Type type, size_t dataSizeInBytes) {
  auto implPtr =
      llvm::makeIntrusiveRefCnt<detail::EvalValueImpl>(type, dataSizeInBytes);
  return EvalValue(implPtr.get());
}

EvalValue Interpreter::createEvalValue(Type type, const void *data,
                                       size_t dataSizeInBytes) {
  auto implPtr = llvm::makeIntrusiveRefCnt<detail::EvalValueImpl>(
    type, llvm::ArrayRef<char>(static_cast<const char *>(data),
                               dataSizeInBytes));
  return EvalValue(implPtr.get());
}
