//===- MLI.cpp - Multi-Level Interpreter Driver -------------*- C++ -*-===//
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
#include "MLIExec.h"
#include "MLIUtils.h"
#include "MLIFormat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interpreter/Dialects/FuncInterpreter.h"
#include "mlir/Interpreter/Dialects/LLVMInterpreter.h"
#include "mlir/Interpreter/Dialects/ArithInterpreter.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {

  mlir::MLIRContext context;

  // Allow unregistered dialects for now until we adequately support all dialects or 
  // the ones that persistently show up in the attributes section
  context.allowUnregisteredDialects();

  // Register command-line options
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input mlir file>"), llvm::cl::Required);
  llvm::cl::opt<std::string> funcName("func", llvm::cl::desc("Specify function entry point"), llvm::cl::value_desc("function"), llvm::cl::init("main"));
  llvm::cl::list<std::string> args("args", llvm::cl::desc("List of numeric arguments"), llvm::cl::CommaSeparated);

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Interpreter Driver\n");

  // Register the necessary dialects
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();

  mlir::Interpreter interpreter(context);

  // Register the necessary interpreters
  interpreter.registerDialectInterpreter<mlir::FuncInterpreter>();
  interpreter.registerDialectInterpreter<mlir::LLVMInterpreter>();
  interpreter.registerDialectInterpreter<mlir::ArithInterpreter>();

  // Parse the MLIR file 
  auto module = mli::parseMLIRFile(inputFilename, context);
  if (!module) {
    llvm::errs() << mli::fmt::error("Failed to parse MLIR file: " + inputFilename) << "\n";
    return 1;
  }
  llvm::outs() << mli::fmt::success("Successfully parsed MLIR file: " + inputFilename) << "\n";
  
  // Set the module in the interpreter
  interpreter.setModule(*module);

  // Get the function
  auto func = module->lookupSymbol<mlir::func::FuncOp>(funcName);
  if (!func) {
    llvm::errs() << "Function '" << funcName << "' not found in the module.\n";
    llvm::errs() << "Available functions: ";
    for (auto &op : module->getOps()) {
      if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
        llvm::errs() << funcOp.getName() << ", ";
      } else {
        llvm::errs() << op.getName() << " is not a func.func operation.\n";
      }
    }
    return 1;
  }

  // Set up the function frame
  mlir::ScopedFunctionFrame functionFrameGuard(interpreter);

  // Set up the region frame (important for handling block arguments and operation execution)
  auto &entryBlock = func.getBody().front();
  mlir::ScopedRegionFrame regionFrameGuard(interpreter);

  // Ensure the number of arguments matches the function signature
  if (entryBlock.getNumArguments() != args.size()) {
    llvm::errs() << "Mismatch between number of provided arguments and function signature.\n";
    return 1;
  }

  // Prepare arguments with correct sizes and types
  mlir::SmallVector<mlir::EvalValue, 4> arguments;
  int i = 0;
  for (auto &arg : args) {
    // Use type from function argument in constructing data
    auto arg_type = entryBlock.getArgument(i).getType();
    unsigned width = arg_type.getIntOrFloatBitWidth();
    if (arg_type.isInteger(width)) {
        int64_t int_val = std::stoll(arg);
        printf("Parsing %s into int with width %u\n", arg.c_str(), width);
        arguments.push_back(interpreter.createEvalValue(mlir::IntegerType::get(&context, width), &int_val, width));
    }
    else if (arg_type.isF32()) {
        float float_val = std::stof(arg);
        printf("Parsing %s into float %f with width %u\n", arg.c_str(), float_val, width);
        arguments.push_back(interpreter.createEvalValue(mlir::Float32Type::get(&context), &float_val, width));
    }
    else if (arg_type.isF64()) {
        double float_val = std::stod(arg);
        printf("Parsing %s into float %f with width %u\n", arg.c_str(), float_val, width);
        arguments.push_back(interpreter.createEvalValue(mlir::Float64Type::get(&context), &float_val, width));
    }
    else {
        llvm::errs() << "Unrecognized type, not int or floating point" << "\n";
    }
    i++;
  }

  // Initialize and map block arguments to EvalValues
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    mlir::Value blockArg = entryBlock.getArgument(i);
    interpreter.setEvalValue(blockArg, arguments[i]);
  }

  mlir::EvalValue lastReturnValue;
  for (auto &op : entryBlock) {
    llvm::SmallVector<mlir::EvalValue, 4> opOperands;
    for (auto operand : op.getOperands()) {
      llvm::outs() << "operand: " << operand << "\n";
      if (auto evalValue = interpreter.getEvalValue(operand)) {
        opOperands.push_back(evalValue);
      } else {
        llvm::errs() << "Operand not found in interpreter context.\n";
        return 1;
      }
    }

    mlir::EvalResult result = interpreter.execute(op, opOperands);

    if (result.getKind() == mlir::EvalResultKind::Error) {
      llvm::errs() << "Error occurred during execution of operation: " << op << "\n";
      llvm::errs() << "Error message: " << result.getError().getMessage() << "\n";
      return 1;
    }

    // Handle return values from any dialect
    if (result.getKind() == mlir::EvalResultKind::ReturnValue) {
      llvm::outs() << mli::fmt::info("Return operation detected: " + op.getName().getStringRef().str()) << "\n";
      auto returnVals = result.getValues();
      if (!returnVals.empty()) {
        lastReturnValue = returnVals.back();
        // Print the return value type
        std::string typeStr;
        llvm::raw_string_ostream typeOs(typeStr);
        lastReturnValue.getType().print(typeOs);
        llvm::outs() << mli::fmt::dim("Return type: ") << mli::fmt::type(typeStr) << "\n";
      } else {
        llvm::outs() << mli::fmt::dim("Void return detected") << "\n";
      }
      break;
    }

    // For non-return operations, bind results to the interpreter context
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto resultValue = op.getResult(i);
      if (i < result.getValues().size()) {
        interpreter.setEvalValue(resultValue, result.getValues()[i]);
      } else {
        llvm::errs() << "Mismatch between operation results and EvalResult values.\n";
        return 1;
      }
    }
  }

  // Print final return status
  if (lastReturnValue) {
    llvm::outs() << mli::fmt::success("Function execution completed with return value") << "\n";
  } else {
    llvm::outs() << mli::fmt::info("Function execution completed (void return)") << "\n";
  }

  return 0;
}
