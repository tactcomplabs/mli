#include "MLIFormat.h"
#include "MLIUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interpreter/Dialects/FuncInterpreter.h"
#include "mlir/Interpreter/Dialects/LLVMInterpreter.h"
#include "mlir/Interpreter/Interpreter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

// Converts a command-line argument to an EvalValue based on the expected type
static mlir::EvalValue convertArgToEvalValue(
    const std::string& argStr,
    mlir::Type argType,
    mlir::Interpreter& interpreter,
    mlir::MLIRContext& context) {

  if (mlir::isa<mlir::IndexType>(argType)) {
    intptr_t idx_val = std::stoll(argStr);
    llvm::outs() << mli::fmt::dim("Parsing ")
                 << mli::fmt::highlight(argStr) << mli::fmt::dim(" as ")
                 << mli::fmt::type("index") << "\n";
    return interpreter.createEvalValue(argType, &idx_val, sizeof(idx_val));
  }
  else if (mlir::isa<mlir::IntegerType>(argType)) {
    unsigned width = argType.getIntOrFloatBitWidth();
    llvm::outs() << mli::fmt::dim("Parsing ")
                << mli::fmt::highlight(argStr) << mli::fmt::dim(" as ")
                << mli::fmt::type("i" + std::to_string(width)) << "\n";
    if (width == 1) {
        bool bool_val = argStr != "0" && argStr != "false";
        return interpreter.createEvalValue(argType, &bool_val, sizeof(bool_val));

    }
    APInt int_val = APInt(width, argStr, 10);
    return interpreter.createEvalValue(argType, &int_val, sizeof(int_val));
  }
  else if (mlir::isa<mlir::FloatType>(argType)) {
    unsigned width = argType.getIntOrFloatBitWidth();
    Semantics s = mli::getFloatSemantics(argType);
    APFloat float_val = APFloat(APFloatBase::EnumToSemantics(s), argStr);
    llvm::outs() << mli::fmt::dim("Parsing ")
                << mli::fmt::highlight(argStr) << mli::fmt::dim(" as ")
                << mli::fmt::type("f" + std::to_string(width)) << "\n";
    return interpreter.createEvalValue(argType, &float_val, sizeof(float_val));
  }

  throw std::runtime_error("Unsupported argument type");
}

// Prepares arguments from command line for function execution
static mlir::SmallVector<mlir::EvalValue, 4> prepareArguments(
    mlir::func::FuncOp& func,
    llvm::cl::list<std::string>& args,
    mlir::Interpreter& interpreter,
    mlir::MLIRContext& context) {

  mlir::SmallVector<mlir::EvalValue, 4> functionArgs;
  auto& entryBlock = func.getBody().front();

  // Convert command-line arguments to EvalValues
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg_type = entryBlock.getArgument(i).getType();

    try {
      functionArgs.push_back(convertArgToEvalValue(args[i], arg_type, interpreter, context));
    } catch (const std::exception& e) {
      llvm::errs() << mli::fmt::error("Failed to parse argument '" + args[i] + "'") << "\n";
      throw; // Re-throw to be caught by caller
    }
  }

  return functionArgs;
}

// Represents a block to be executed with its arguments
struct BlockExecution {
  mlir::Block* block;
  mlir::SmallVector<mlir::EvalValue, 4> args;
};

// Processes the result of a function execution for display
static void processExecutionResult(const mlir::EvalResult& result) {
  if (result.getKind() == mlir::EvalResultKind::ReturnValue) {
    auto returnVals = result.getValues();
    if (!returnVals.empty()) {
      // Get the type
      llvm::SmallVector<mlir::Type, 4> returnTypes;
      for (const auto& val: returnVals) {
        returnTypes.push_back(val.getType());
      }
      std::string typeStr = mli::printAsList(llvm::ArrayRef(returnTypes));

      // Format the value
      std::string valueStr = mli::printAsList(returnVals);

      llvm::outs() << mli::fmt::success("Function returned") << " "
                  << mli::fmt::type(typeStr) << " = " << mli::fmt::highlight(valueStr) << "\n";
    } else {
      llvm::outs() << mli::fmt::success("Function completed") << " "
                  << mli::fmt::type("void") << "\n";
    }
  } else {
    // If we didn't get a ReturnValue, something unusual happened
    llvm::errs() << mli::fmt::warning("Function execution completed without proper return")
                << "\n";
  }
}

// Executes a single operation and handles its result
static std::pair<mlir::EvalResult, bool> executeOperation(
    mlir::Operation& op,
    mlir::Interpreter& interpreter,
    std::vector<BlockExecution>& executionStack) {

  // Get operands
  llvm::SmallVector<mlir::EvalValue, 4> opOperands;

  for (auto operand : op.getOperands()) {
    if (auto evalValue = interpreter.getEvalValue(operand)) {
      opOperands.push_back(evalValue);
    } else {
      return {
        interpreter.createErrorResult("Missing operand value for operation"),
        true // hasCompleted
      };
    }
  }

  // Execute operation
  mlir::EvalResult opResult = interpreter.execute(op, opOperands);

  if (opResult.getKind() == mlir::EvalResultKind::Error) {
    return {std::move(opResult), true}; // hasCompleted
  }
  else if (opResult.getKind() == mlir::EvalResultKind::ReturnValue ||
           opResult.getKind() == mlir::EvalResultKind::YieldValue) {
    return {std::move(opResult), true}; // hasCompleted
  }
  else if (opResult.getKind() == mlir::EvalResultKind::Branch) {
    // Push the destination block onto our execution stack
    if (opResult.getBlock()) {
      executionStack.push_back({
        opResult.getBlock(),
        mlir::SmallVector<mlir::EvalValue, 4>(
            opResult.getValues().begin(),
            opResult.getValues().end())
      });
      return {mlir::EvalResult(), false}; // Signal continue with no final result yet
    } else {
      return {
        interpreter.createErrorResult("Branch to null block"),
        true // hasCompleted
      };
    }
  }
  else if (opResult.getKind() == mlir::EvalResultKind::BindValue) {
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      if (i < opResult.getValues().size()) {
        interpreter.setEvalValue(op.getResult(i), opResult.getValues()[i]);
      }
    }
    return {mlir::EvalResult(), false}; // Signal continue with no final result yet
  }

  // Default case
  return {mlir::EvalResult(), false};
}

// Executes a single block and returns the result
static mlir::EvalResult executeBlock(
    BlockExecution& blockExec,
    mlir::Interpreter& interpreter,
    std::vector<BlockExecution>& executionStack) {

  // Bind arguments
  if (blockExec.block->getNumArguments() != blockExec.args.size()) {
    return interpreter.createErrorResult(
        "Block argument count mismatch: expected " +
        std::to_string(blockExec.block->getNumArguments()) +
        ", got " + std::to_string(blockExec.args.size()));
  }

  for (unsigned i = 0; i < blockExec.block->getNumArguments(); ++i) {
    interpreter.setEvalValue(blockExec.block->getArgument(i), blockExec.args[i]);
  }

  // Execute operations
  for (auto& op : *blockExec.block) {
    auto [opResult, hasCompleted] = executeOperation(op, interpreter, executionStack);

    if (hasCompleted) {
      return std::move(opResult);
    }

    // If not completed but we were told to stop this block, break out
    if (!executionStack.empty() && executionStack.back().block != blockExec.block) {
      break;
    }
  }

  // If we reach here, the block ended without a terminator
  return interpreter.createErrorResult("Block ended without terminator");
}

// Main entry point for executing a function
static int executeFunction(
    mlir::Interpreter& interpreter,
    mlir::func::FuncOp func,
    llvm::cl::list<std::string>& args,
    mlir::MLIRContext& context) {

  try {
    // Prepare arguments
    mlir::SmallVector<mlir::EvalValue, 4> functionArgs =
        prepareArguments(func, args, interpreter, context);

    // Create scoped frames for execution
    mlir::ScopedFunctionFrame functionFrame(interpreter);
    mlir::ScopedRegionFrame regionFrame(interpreter);

    // Setup execution stack
    std::vector<BlockExecution> executionStack;
    executionStack.push_back({&func.getBody().front(), functionArgs});

    mlir::EvalResult finalResult;
    bool hasCompleted = false;

    // Iterative execution loop
    while (!executionStack.empty() && !hasCompleted) {
      // Get the next block to execute
      BlockExecution current = executionStack.back();
      executionStack.pop_back();

      // Execute the block
      mlir::EvalResult blockResult = executeBlock(current, interpreter, executionStack);

      // If the block execution returned a result, we're done
      if (blockResult.getKind() != mlir::EvalResultKind::Error || executionStack.empty()) {
        finalResult = std::move(blockResult);
        hasCompleted = true;
      }
    }

    // Process errors
    if (finalResult.getKind() == mlir::EvalResultKind::Error) {
      llvm::errs() << mli::fmt::error("Error in function execution: " +
                                    finalResult.getError().getMessage()) << "\n";
      return 1;
    }

    // Process the final result for display
    processExecutionResult(finalResult);

    return 0;

  } catch (const std::exception& e) {
    // Handle any exceptions from argument parsing or other issues
    llvm::errs() << mli::fmt::error(std::string("Exception during function execution: ") + e.what()) << "\n";
    return 1;
  }
}


