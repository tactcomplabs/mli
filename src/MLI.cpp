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

int main(int argc, char **argv) {

  mlir::MLIRContext context;

  // Allow unregistered dialects for now until we adequately support all dialects or 
  // the ones that persistently show up in the attributes section
  context.allowUnregisteredDialects();

  // Register command-line options
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input mlir file>"), llvm::cl::Required);
  llvm::cl::opt<std::string> funcName("func", llvm::cl::desc("Specify function entry point"), llvm::cl::value_desc("function"), llvm::cl::init("main"));
  llvm::cl::list<std::string> args("args", llvm::cl::desc("List of numeric arguments"), llvm::cl::CommaSeparated);
  static llvm::cl::opt<bool, true> printFlag("pretty-print", llvm::cl::desc("Enable ANSI pretty output"), llvm::cl::location(mli::fmt::usePrettyPrint), llvm::cl::init(true));

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
    return 1;
  }
  
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

  return executeFunction(interpreter, func, args, context);

//   // Set up the function frame
//   mlir::ScopedFunctionFrame functionFrameGuard(interpreter);

//   // Set up the region frame (important for handling block arguments and operation execution)
//   auto &entryBlock = func.getBody().front();
//   mlir::ScopedRegionFrame regionFrameGuard(interpreter);

//   // Ensure the number of arguments matches the function signature
//   if (entryBlock.getNumArguments() != args.size()) {
//     llvm::errs() << mli::fmt::error("Mismatch between number of provided arguments and function signature.") << "\n";
//     return 1;
//   }

//   // Initialize and map block arguments to EvalValues
//   for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
//     mlir::Value blockArg = entryBlock.getArgument(i);
//     mlir::EvalValue argVal = convertArgToEvalValue(args[i], blockArg.getType(), interpreter, context);
//     interpreter.setEvalValue(blockArg, argVal);
//   }

//   bool isVoidReturn = false;
//   for (auto &op : entryBlock) {
//     llvm::SmallVector<mlir::EvalValue, 4> opOperands;
//     for (auto operand : op.getOperands()) {
//       if (auto evalValue = interpreter.getEvalValue(operand)) {
//         opOperands.push_back(evalValue);
//       } else {
//         llvm::errs() << mli::fmt::error("Operand not found in interpreter context") << "\n";
//         return 1;
//       }
//     }

//     mlir::EvalResult result = interpreter.execute(op, opOperands);

//     if (result.getKind() == mlir::EvalResultKind::Error) {
//       llvm::errs() << mli::fmt::error("Error occurred during execution of operation") << op << "\n";
//       llvm::errs() << "Error message: " << result.getError().getMessage() << "\n";
//       return 1;
//     }

//     // Handle return values from any dialect
//     if (result.getKind() == mlir::EvalResultKind::ReturnValue) {
//       llvm::outs() << mli::fmt::info("Return operation detected: " + op.getName().getStringRef().str()) << "\n";
//       auto returnVals = result.getValues();
//       isVoidReturn = returnVals.empty();
//       if (isVoidReturn) {
//         llvm::outs() << mli::fmt::dim("Void return detected") << "\n";
//         break;
//       }
//       // Print the return value type
//       std::string typeStr;
//       llvm::raw_string_ostream typeOs(typeStr);
//       for (auto it = returnVals.begin(); it != returnVals.end(); it++) {
//         if (it != returnVals.begin()) typeOs << ", ";
//         it->getType().print(typeOs);
//       }
//       llvm::outs() << mli::fmt::dim("Return type: ") << mli::fmt::type(typeStr) << "\n";
//     }

//     // For non-return operations, bind results to the interpreter context
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       auto resultValue = op.getResult(i);
//       if (i < result.getValues().size()) {
//         interpreter.setEvalValue(resultValue, result.getValues()[i]);
//       } else {
//         llvm::errs() << "Mismatch between operation results and EvalResult values.\n";
//         return 1;
//       }
//     }
//   }

//   // Print final return status
//   if (isVoidReturn) {
//     llvm::outs() << mli::fmt::success("Function execution completed (void return)") << "\n";
//   } else {
//     llvm::outs() << mli::fmt::success("Function execution completed with return value") << "\n";
//   }

  return 0;
}
