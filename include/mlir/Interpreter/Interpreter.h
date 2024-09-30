//===- Interpreter.h - MLIR Interpreter API ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Interpreter class and the APIs of the interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERPRETER_INTERPRETER_H_
#define MLIR_INTERPRETER_INTERPRETER_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <forward_list>
#include <utility>

namespace mlir {

namespace detail {

// Owning pointer for type-erased dialect op interpreter context.
class OwningDialectInterpreterContext {
public:
  OwningDialectInterpreterContext() : context(nullptr), deleter(nullptr) {}
  OwningDialectInterpreterContext(void *context, void (*deleter)(void *))
      : context(context), deleter(deleter) {}
  OwningDialectInterpreterContext(OwningDialectInterpreterContext &&rhs)
      : context(rhs.context), deleter(rhs.deleter) {
    rhs.context = nullptr;
    rhs.deleter = nullptr;
  }
  OwningDialectInterpreterContext &
  operator=(OwningDialectInterpreterContext &&rhs) {
    if (&rhs != this) {
      context = rhs.context;
      deleter = rhs.deleter;
      rhs.context = nullptr;
      rhs.deleter = nullptr;
    }
    return *this;
  }
  ~OwningDialectInterpreterContext() {
    if (context) {
      deleter(context);
    }
  }

  void *get() const { return context; }

private:
  void *context;
  void (*deleter)(void *);

private:
  OwningDialectInterpreterContext(const OwningDialectInterpreterContext &) =
      delete;
  OwningDialectInterpreterContext &
  operator=(const OwningDialectInterpreterContext &rhs) = delete;
};

} // namespace detail

class Interpreter {
public:
  explicit Interpreter(MLIRContext &context,
                       bool enableStackTraceOnError = false);

  //===--------------------------------------------------------------------===//
  // Dialect interpreter registration
  //===--------------------------------------------------------------------===//

  /// Register a dialect op interpreter.
  template <typename DialectInterpreter, typename... Args>
  void registerDialectInterpreter(Args &&...args) {
    using Dialect = typename DialectInterpreter::Dialect;

    // Register and load the dialect.
    DialectRegistry dialects;
    dialects.insert<Dialect>();
    context->appendDialectRegistry(dialects);
    context->loadDialect<Dialect>();

    DialectInterpreter interpreterInstance;
    interpreterInstance.attachInterface(*context);

    // Create dialect contexts.
    detail::OwningDialectInterpreterContext context(
        DialectInterpreter::createContext(std::forward<Args>(args)...),
        [](void *context) {
          using Context = typename DialectInterpreter::Context;
          DialectInterpreter::destroyContext(
              reinterpret_cast<Context *>(context));
        });
    dialectContexts[Dialect::getDialectNamespace()] = std::move(context);
  }

  template <typename DialectInterpreter, typename OtherDialectInterpreter,
            typename... MoreDialectInterpreters, typename... Args>
  void registerDialectInterpreter(Args &&...args) {
    registerDialectInterpreter<DialectInterpreter>();
    registerDialectInterpreter<OtherDialectInterpreter,
                               MoreDialectInterpreters...>(
        std::forward<Args>(args)...);
  }

  /// Get the dialect interpreter context.
  template <typename DialectInterpreter>
  typename DialectInterpreter::Context *getDialectInterpreterContext() const {
    using Dialect = typename DialectInterpreter::Dialect;
    auto iter = dialectContexts.find(Dialect::getDialectNamespace());
    if (iter == dialectContexts.end()) {
      return nullptr;
    }
    return reinterpret_cast<typename DialectInterpreter::Context *>(
        iter->second.get());
  }

  //===--------------------------------------------------------------------===//
  // MLIRContext and Module accessors
  //===--------------------------------------------------------------------===//

  /// Get the MLIR context.
  MLIRContext &getContext() { return *context; }

  /// Get the ModuleOp.
  ModuleOp getModule() { return module; }

  /// Set the ModuleOp.
  void setModule(ModuleOp newModule) { module = newModule; }

  //===--------------------------------------------------------------------===//
  // Execution functions
  //===--------------------------------------------------------------------===//

  /// Find the function by name and execute the function with the function
  /// arguments.
  EvalResult execute(StringRef entry_func_name, ArrayRef<EvalValue> arguments);

  /// Execute a function with the function arguments.
  EvalResult execute(func::FuncOp func, ArrayRef<EvalValue> arguments);

  /// Function argument provider callback function.
  ///
  /// All `execute` methods for `func::FuncOp` (or function name) has a variant
  /// that takes `ArgumentProviderRef`. These `execute` methods pass the
  /// function argument types to `argument_provider` and use the returned values
  /// as actual arguments. With this interface, the callers of `execute` don't
  /// have to find the `func::FuncOp` and extract its argument types by
  /// themselves.
  using ArgumentProviderRef =
      llvm::function_ref<EvalResult(Interpreter &, TypeRange)>;

  /// Find the function by name and execute the function with the function
  /// arguments from the argument provider.
  EvalResult execute(StringRef entry_func_name,
                     ArgumentProviderRef argument_provider);

  /// Execute a function with the function arguments from the argument provider.
  EvalResult execute(func::FuncOp func, ArgumentProviderRef argument_provider);

  /// Execute a region with the block arguments of the entry block.
  EvalResult execute(Region &region, ArrayRef<EvalValue> arguments);

  /// Execute a block with the block arguments.
  EvalResult execute(Block &block, ArrayRef<EvalValue> arguments);

  /// Execute an operation (get the input operands from the
  /// SSA-name-to-evaluated-value map).
  EvalResult execute(Operation &operation);

  /// Execute an operation with the input operands.
  EvalResult execute(Operation &operation, ArrayRef<EvalValue> operands);

  //===--------------------------------------------------------------------===//
  // Interpreter frame management functions
  //===--------------------------------------------------------------------===//

  /// Push a function frame (SSA lookup only can search up to the function
  /// frame. The callee must not refer the SSA defined by the caller.)
  void pushFunctionFrame() {
    evalValueMapStack.push_front(std::forward_list<EvalValueMap>());
  }

  /// Pop a function frame.
  void popFunctionFrame() {
    assert(!evalValueMapStack.empty() && "function frame must be available");
    assert(evalValueMapStack.front().empty() &&
           "all region frames must be popped");
    evalValueMapStack.pop_front();
  }

  /// Push a region frame (SSA lookup can search the outer region frame up
  /// until the function frame).
  void pushRegionFrame() {
    assert(!evalValueMapStack.empty() && "function frame must be available");
    evalValueMapStack.front().push_front(EvalValueMap());
  }

  /// Pop a region frame.
  void popRegionFrame() {
    assert(!evalValueMapStack.empty() && "function frame must be available");
    assert(!evalValueMapStack.front().empty() &&
           "region frame must be available");
    evalValueMapStack.front().pop_front();
  }

  /// Get the evaluated value of an SSA name.
  EvalValue getEvalValue(Value ssaName) const {
    assert(!evalValueMapStack.empty() && "function frame must be available");
    for (auto stackIter = evalValueMapStack.front().begin();
         stackIter != evalValueMapStack.front().end(); ++stackIter) {
      auto iter = stackIter->find(ssaName);
      if (iter != stackIter->end()) {
        return iter->second;
      }
    }
    return EvalValue();
  }

  /// Bind an SSA name to an evaluated value.
  void setEvalValue(Value ssaName, EvalValue value) {
    assert(!evalValueMapStack.empty() && "function frame must be available");
    assert(!evalValueMapStack.front().empty() &&
           "region frame must be available");
    evalValueMapStack.front().front()[ssaName] = std::move(value);
  }

  /// Bind SSA names to evaluated values. Returns failure if the number of the
  /// SSA names and the number of the evaluated values mismatches.
  LogicalResult setEvalValues(ValueRange ssaNames,
                              ArrayRef<EvalValue> evalValues);

  //===--------------------------------------------------------------------===//
  // EvalResult functions
  //===--------------------------------------------------------------------===//

  /// Create an EvalResult for an error message.
  EvalResult createErrorResult(StringRef errorMessage);

  /// Create an EvalResult to bind op results to values.
  EvalResult createBindValueResult(ArrayRef<EvalValue> values) {
    return EvalResult(EvalResultKind::BindValue, values, nullptr, nullptr);
  }

  /// Create an EvalResult for returning values.
  EvalResult createReturnValueResult(ArrayRef<EvalValue> values) {
    return EvalResult(EvalResultKind::ReturnValue, values, nullptr, nullptr);
  }

  /// Create an EvalResult for yielding values.
  EvalResult createYieldValueResult(ArrayRef<EvalValue> values) {
    return EvalResult(EvalResultKind::YieldValue, values, nullptr, nullptr);
  }

  /// Create an EvalResult for branching to another block.
  ///
  /// Explanation:
  /// br ^bb1(%1 : i32)
  /// ^bb1(%arg1: i32):
  ///   %2 = addi %arg1, %0 : i32
  ///   ...
  /// Explanation:
  /// The br operation branches to the block ^bb1, passing %1 as an argument.
  /// The interpreter would produce an EvalResultKind::Branch as it transitions to the new block with %1 as the block argument.
  EvalResult createBranchResult(Block &destBlock, ArrayRef<EvalValue> values) {
    return EvalResult(EvalResultKind::Branch, values, &destBlock, nullptr);
  }

  //===--------------------------------------------------------------------===//
  // EvalValue functions
  //===--------------------------------------------------------------------===//

  /// Create an EvalValue with uninitialized data buffer.
  EvalValue createEvalValue(Type type, size_t dataSizeInBytes);

  /// Create an EvalValue and initialize with `data`.
  EvalValue createEvalValue(Type type, const void *data,
                            size_t dataSizeInBytes);

  /// Create an EvalValue and initialize with `data`.
  template <typename T>
  EvalValue createEvalValue(Type type, llvm::ArrayRef<T> data) {
    return createEvalValue(type, reinterpret_cast<const char *>(data.data()),
                           sizeof(T) * data.size());
  }

private:
  /// Mapping from SSA names to evaluated value. This represents a value lookup
  /// scope within a region.
  using EvalValueMap = llvm::DenseMap<Value, EvalValue>;

  /// Mapping from dialect names to their interpreter context.
  using DialectInterpreterContextMap =
      llvm::DenseMap<llvm::StringRef, detail::OwningDialectInterpreterContext>;

  /// Mapping from dialect names to dialect interpreter contexts.
  DialectInterpreterContextMap dialectContexts;

  /// MLIRContext for this interpreter.
  MLIRContext *context;

  /// ModuleOp being interpreted. Typically, the interpreter will search the
  /// callee function within this module.
  ModuleOp module;

  /// Stack for the mappings from SSA name to evaluated value.
  std::forward_list<std::forward_list<EvalValueMap>> evalValueMapStack;

  /// Whether the interpreter should include a stack trace in `EvalError`.
  bool enableStackTraceOnError;

private:
  Interpreter(const Interpreter &) = delete;
  Interpreter &operator=(const Interpreter &) = delete;
};

/// Helper class to push and pop a function frame in a C++ scope.
class ScopedFunctionFrame {
private:
  Interpreter &interpreter;

public:
  explicit ScopedFunctionFrame(Interpreter &interpreter)
      : interpreter(interpreter) {
    interpreter.pushFunctionFrame();
  }

  ~ScopedFunctionFrame() { interpreter.popFunctionFrame(); }
};

/// Helper class to push and pop a region frame in a C++ scope.
class ScopedRegionFrame {
private:
  Interpreter &interpreter;

public:
  explicit ScopedRegionFrame(Interpreter &interpreter)
      : interpreter(interpreter) {
    interpreter.pushRegionFrame();
  }

  ~ScopedRegionFrame() { interpreter.popRegionFrame(); }
};

} // namespace mlir

#endif // MLIR_INTERPRETER_INTERPRETER_H_
