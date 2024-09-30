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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVM/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interpreter/Dialects/FuncInterpreter.h"
#include "mlir/Interpreter/Dialects/LLVMInterpreter.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  mlir::MLIRContext context;

  // Register command-line options
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input mlir file>"), llvm::cl::Required);
  llvm::cl::opt<std::string> funcName("func", llvm::cl::desc("Specify function entry point"), llvm::cl::value_desc("function"), llvm::cl::init("main"));
  llvm::cl::list<int32_t> args("args", llvm::cl::desc("List of integer arguments"), llvm::cl::CommaSeparated);

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Interpreter Driver\n");

  // Register the necessary dialects
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  // Parse the MLIR file
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "Failed to parse MLIR file: " << inputFilename << "\n";
    return 1;
  }

  mlir::Interpreter interpreter(context);

  // Register the necessary interpreters
  interpreter.registerDialectInterpreter<mlir::FuncInterpreter>();
  interpreter.registerDialectInterpreter<mlir::LLVMInterpreter>();

  // Set the module in the interpreter
  interpreter.setModule(*module);

  // Prepare arguments with correct size
  mlir::SmallVector<mlir::EvalValue, 4> arguments;
  for (auto &arg : args) {
    arguments.push_back(interpreter.createEvalValue(mlir::IntegerType::get(&context, 32), &arg, sizeof(arg)));
  }

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
  if (entryBlock.getNumArguments() != arguments.size()) {
    llvm::errs() << "Mismatch between number of provided arguments and function signature.\n";
    return 1;
  }

  // Initialize and map block arguments to EvalValues
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    mlir::Value blockArg = entryBlock.getArgument(i);
    interpreter.setEvalValue(blockArg, arguments[i]);
  }

  const char* retval;
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
      return 1;
    }

    unsigned i = 0;
    for (i = 0; i < op.getNumResults(); ++i) {
      auto resultValue = op.getResult(i);
      if (i < result.getValues().size()) {
        interpreter.setEvalValue(resultValue, result.getValues()[i]);
      } else {
        llvm::errs() << "Mismatch between operation results and EvalResult values.\n";
        return 1;
      }
    }

    if (llvm::isa<mlir::LLVM::ReturnOp>(op)) {
      llvm::outs() << "Returning from function.\n";
      // save the last result
      // the interpreter will return the last result
      retval = result.getValues().back().getRawData();
      break;
    }
  }

  // Look for the value of the variable being returned
  if (retval) {
    llvm::outs() << "ret: " << retval << "\n";
  } else {
    llvm::errs() << "No return values produced by function execution. UH OH!\n";
  }

  return 0;
}
