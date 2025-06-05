#include "MLIFormat.h"
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
  
  unsigned width = argType.getIntOrFloatBitWidth();
  
  if (argType.isInteger()) {
    int64_t int_val = std::stoll(argStr);
    llvm::outs() << mli::fmt::dim("Parsing ")
                << mli::fmt::highlight(argStr) << mli::fmt::dim(" as ")
                << mli::fmt::type("i" + std::to_string(width)) << "\n";
    return interpreter.createEvalValue(
        mlir::IntegerType::get(&context, width), &int_val, sizeof(int_val));
  } 
  else if (argType.isF32()) {
    float float_val = std::stof(argStr);
    llvm::outs() << mli::fmt::dim("Parsing ")
                << mli::fmt::highlight(argStr) << mli::fmt::dim(" as ")
                << mli::fmt::type("f32") << "\n";
    return interpreter.createEvalValue(
        mlir::Float32Type::get(&context), &float_val, sizeof(float_val));
  } 
  else if (argType.isF64()) {
    double float_val = std::stod(argStr);
    llvm::outs() << mli::fmt::dim("Parsing ")
                << mli::fmt::highlight(argStr) << mli::fmt::dim(" as ")
                << mli::fmt::type("f64") << "\n";
    return interpreter.createEvalValue(
        mlir::Float64Type::get(&context), &float_val, sizeof(float_val));
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

// Formats a return value as a string for display
static std::string formatReturnValue(const mlir::EvalValue& value) {
  std::string valueStr;
  llvm::raw_string_ostream valueOs(valueStr);
  
  auto valType = value.getType();
  
  // Handle integer types
  if (valType.isIntOrIndex()) {
    if (valType.getIntOrFloatBitWidth() <= 32) {
      auto data = value.getData<int32_t>();
      if (!data.empty()) {
        valueOs << data[0];
      }
    } else {
      auto data = value.getData<int64_t>();
      if (!data.empty()) {
        valueOs << data[0];
      }
    }
  }
  // Handle float types
  else if (valType.isF32()) {
    auto data = value.getData<float>();
    if (!data.empty()) {
      // Format float with fixed precision
      char buffer[32];
      snprintf(buffer, sizeof(buffer), "%.6f", data[0]);
      valueOs << buffer;
    }
  } 
  else if (valType.isF64()) {
    auto data = value.getData<double>();
    if (!data.empty()) {
      // Format double with fixed precision
      char buffer[32];
      snprintf(buffer, sizeof(buffer), "%.6f", data[0]);
      valueOs << buffer;
    }
  }
  // Handle pointer types
  else if (mlir::isa<mlir::LLVM::LLVMPointerType>(valType)) {
    auto data = value.getData<uint64_t>();
    if (!data.empty()) {
      valueOs << "0x";
      valueOs.write_hex(data[0]);
    }
  }
  // Default fallback - show raw bytes in hex
  else {
    const char* rawData = value.getRawData();
    size_t dataSize = value.getRawDataSizeInBytes();
    
    valueOs << "0x";
    for (size_t i = 0; i < dataSize; i++) {
      // Format each byte as hex
      char buffer[3];
      snprintf(buffer, sizeof(buffer), "%02x", (unsigned char)rawData[i]);
      valueOs << buffer;
    }
  }
  
  valueOs.flush();
  return valueStr;
}

// Processes the result of a function execution for display
static void processExecutionResult(const mlir::EvalResult& result) {
  if (result.getKind() == mlir::EvalResultKind::ReturnValue) {
    auto returnVals = result.getValues();
    if (!returnVals.empty()) {
      // Get the type
      std::string typeStr;
      llvm::raw_string_ostream typeOs(typeStr);
      returnVals.back().getType().print(typeOs);
      typeOs.flush();
      
      // Format the value
      std::string valueStr = formatReturnValue(returnVals.back());
      
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


