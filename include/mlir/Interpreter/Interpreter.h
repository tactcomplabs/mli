//===- LLVMInterpreter.h - LLVM dialect interpreter -------------*- C++ -*-===//
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
#include "mlir/Interpreter/MemoryManager.h"
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

    OwningDialectInterpreterContext(void* context, void (*deleter)(void*)) : context(context), deleter(deleter) {}

    OwningDialectInterpreterContext(OwningDialectInterpreterContext&& rhs) : context(rhs.context), deleter(rhs.deleter) {
        rhs.context = nullptr;
        rhs.deleter = nullptr;
    }

    OwningDialectInterpreterContext& operator=(OwningDialectInterpreterContext&& rhs) {
        if ( &rhs != this ) {
            context     = rhs.context;
            deleter     = rhs.deleter;
            rhs.context = nullptr;
            rhs.deleter = nullptr;
        }
        return *this;
    }

    ~OwningDialectInterpreterContext() {
        if ( context ) {
            deleter(context);
        }
    }

    void* get() const { return context; }

  private:
    void* context;
    void (*deleter)(void*);

  private:
    OwningDialectInterpreterContext(const OwningDialectInterpreterContext&)                = delete;
    OwningDialectInterpreterContext& operator=(const OwningDialectInterpreterContext& rhs) = delete;
};

}  // namespace detail

class Interpreter {
  public:
    explicit Interpreter(
        MLIRContext& context, bool enableStackTraceOnError = false, std::unique_ptr<MemoryManager> MemManager = nullptr
    );

    //===--------------------------------------------------------------------===//
    // Dialect interpreter registration
    //===--------------------------------------------------------------------===//

    /// Register a dialect op interpreter.
    template<typename DialectInterpreter, typename... Args>
    void registerDialectInterpreter(Args&&... args) {
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
            DialectInterpreter::createContext(std::forward<Args>(args)...), [](void* context) {
                using Context = typename DialectInterpreter::Context;
                DialectInterpreter::destroyContext(reinterpret_cast<Context*>(context));
            }
        );
        dialectContexts[Dialect::getDialectNamespace()] = std::move(context);
    }

    template<typename DialectInterpreter, typename OtherDialectInterpreter, typename... MoreDialectInterpreters, typename... Args>
    void registerDialectInterpreter(Args&&... args) {
        registerDialectInterpreter<DialectInterpreter>();
        registerDialectInterpreter<OtherDialectInterpreter, MoreDialectInterpreters...>(std::forward<Args>(args)...);
    }

    /// Get the dialect interpreter context.
    template<typename DialectInterpreter>
    typename DialectInterpreter::Context* getDialectInterpreterContext() const {
        using Dialect = typename DialectInterpreter::Dialect;
        auto iter     = dialectContexts.find(Dialect::getDialectNamespace());
        if ( iter == dialectContexts.end() ) {
            return nullptr;
        }
        return reinterpret_cast<typename DialectInterpreter::Context*>(iter->second.get());
    }

    //===--------------------------------------------------------------------===//
    // MLIRContext and Module accessors
    //===--------------------------------------------------------------------===//

    /// Get the MLIR context.
    MLIRContext* getContext() { return context; }

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

    /// Create an EvalResult for void results.
    EvalResult createVoidResult() { return EvalResult(EvalResultKind::Void, ArrayRef<EvalValue>(), nullptr, nullptr); }

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
    EvalResult createBranchResult(Block& destBlock, ArrayRef<EvalValue> values) {
        return EvalResult(EvalResultKind::Branch, values, &destBlock, nullptr);
    }

    /// Function argument provider callback function.
    ///
    /// All `execute` methods for `func::FuncOp` (or function name) has a variant
    /// that takes `ArgumentProviderRef`. These `execute` methods pass the
    /// function argument types to `argument_provider` and use the returned values
    /// as actual arguments. With this interface, the callers of `execute` don't
    /// have to find the `func::FuncOp` and extract its argument types by
    /// themselves.
    using ArgumentProviderRef = llvm::function_ref<EvalResult(Interpreter&, TypeRange)>;

    /// Find the function by name and execute the function with the function
    /// arguments from the argument provider.
    EvalResult execute(StringRef entry_func_name, ArgumentProviderRef argument_provider);

    /// Execute a function with the function arguments from the argument provider.
    EvalResult execute(func::FuncOp func, ArgumentProviderRef argument_provider);

    /// Execute a region with the block arguments of the entry block.
    EvalResult execute(Region& region, ArrayRef<EvalValue> arguments);

    /// Execute a block with the block arguments.
    EvalResult execute(Block& block, ArrayRef<EvalValue> arguments);

    /// Execute an operation (get the input operands from the
    /// SSA-name-to-evaluated-value map).
    EvalResult execute(Operation& operation);

    /// Execute an operation with the input operands.
    EvalResult execute(Operation& operation, ArrayRef<EvalValue> operands);

    //===--------------------------------------------------------------------===//
    // Interpreter frame management functions
    //===--------------------------------------------------------------------===//

    /// Push a function frame (SSA lookup only can search up to the function
    /// frame. The callee must not refer the SSA defined by the caller.)
    void pushFunctionFrame() { evalValueMapStack.push_front(std::forward_list<EvalValueMap>()); }

    /// Pop a function frame.
    void popFunctionFrame() {
        assert(!evalValueMapStack.empty() && "function frame must be available");
        assert(evalValueMapStack.front().empty() && "all region frames must be popped");
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
        assert(!evalValueMapStack.front().empty() && "region frame must be available");
        evalValueMapStack.front().pop_front();
    }

    /// Get the evaluated value of an SSA name.
    EvalValue getEvalValue(Value ssaName) const {
        assert(!evalValueMapStack.empty() && "function frame must be available");
        for ( auto stackIter = evalValueMapStack.front().begin(); stackIter != evalValueMapStack.front().end(); ++stackIter ) {
            auto iter = stackIter->find(ssaName);
            if ( iter != stackIter->end() ) {
                return iter->second;
            }
        }
        return EvalValue();
    }

    /// Bind an SSA name to an evaluated value.
    void setEvalValue(Value ssaName, EvalValue value) {
        assert(!evalValueMapStack.empty() && "function frame must be available");
        assert(!evalValueMapStack.front().empty() && "region frame must be available");
        evalValueMapStack.front().front()[ssaName] = std::move(value);
    }

    /// Bind SSA names to evaluated values. Returns failure if the number of the
    /// SSA names and the number of the evaluated values mismatches.
    LogicalResult setEvalValues(ValueRange ssaNames, ArrayRef<EvalValue> evalValues);

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

    //===--------------------------------------------------------------------===//
    // EvalValue functions
    //===--------------------------------------------------------------------===//

    /// Create an EvalValue with uninitialized data buffer.
    EvalValue createEvalValue(Type type, size_t dataSizeInBytes);

    /// Create an EvalValue and initialize with `data`.
    template<typename T>
    EvalValue createEvalValue(Type type, const T* data, size_t dataSizeInBytes) {
        if constexpr ( std::is_same_v<T, llvm::APInt> || std::is_same_v<T, llvm::APSInt> ) {
            if ( type.getIntOrFloatBitWidth() > 64 ) {
                auto implPtr = llvm::makeIntrusiveRefCnt<detail::EvalValueImpl>(
                    type,
                    llvm::ArrayRef<char>(reinterpret_cast<const char*>(data->getRawData()), sizeof(uint64_t) * data->getNumWords())
                );
                return EvalValue(implPtr.get());
            }
        }
        else if constexpr ( std::is_same_v<T, llvm::APFloat> ) {
            if ( type.getIntOrFloatBitWidth() > 64 ) {
                APInt bits    = data->bitcastToAPInt();
                auto  implPtr = llvm::makeIntrusiveRefCnt<detail::EvalValueImpl>(
                    type,
                    llvm::ArrayRef<char>(reinterpret_cast<const char*>(bits.getRawData()), sizeof(uint64_t) * bits.getNumWords())
                );
                return EvalValue(implPtr.get());
            }
        }
        auto implPtr = llvm::makeIntrusiveRefCnt<detail::EvalValueImpl>(
            type, llvm::ArrayRef<char>(reinterpret_cast<const char*>(data), dataSizeInBytes)
        );
        return EvalValue(implPtr.get());
    }

    /// Create an EvalValue and initialize with `data`.
    template<typename T>
    EvalValue createEvalValue(Type type, llvm::ArrayRef<T> data) {
        return createEvalValue(type, reinterpret_cast<const char*>(data.data()), sizeof(T) * data.size());
    }

    // TODO: Change to smart pointer
    MemoryManager& getMemManager() { return *MemManager; }

    // Wrappers for MemoryManager's operations
    uint64_t allocateInMemManager(const size_t size) { return MemManager->allocate(size); }

    void freeInMemManager(const uint64_t addr) { MemManager->free(addr); }

    void readFromMemManager(const uint64_t addr, void* dst, const size_t size) const { MemManager->read(addr, dst, size); }

    void writeToMemManager(const uint64_t addr, const void* src, const size_t size) const { MemManager->write(addr, src, size); }

    void forceWriteToMemManager(const uint64_t addr, const void* src, const size_t size) const {
        MemManager->force_write(addr, src, size);
    }

    void copyInMemManager(const uint64_t src, const uint64_t dst, const size_t size) const { MemManager->copy(src, dst, size); }

  private:
    /// Mapping from SSA names to evaluated value. This represents a value lookup
    /// scope within a region.
    using EvalValueMap                 = llvm::DenseMap<Value, EvalValue>;

    /// Mapping from dialect names to their interpreter context.
    using DialectInterpreterContextMap = llvm::DenseMap<llvm::StringRef, detail::OwningDialectInterpreterContext>;

    /// Mapping from dialect names to dialect interpreter contexts.
    DialectInterpreterContextMap dialectContexts;

    /// MLIRContext for this interpreter.
    MLIRContext* context;

    /// ModuleOp being interpreted. Typically, the interpreter will search the
    /// callee function within this module.
    ModuleOp module;

    /// Stack for the mappings from SSA name to evaluated value.
    std::forward_list<std::forward_list<EvalValueMap>> evalValueMapStack;

    /// Whether the interpreter should include a stack trace in `EvalError`.
    bool enableStackTraceOnError;

  private:
    Interpreter(const Interpreter&)                              = delete;
    Interpreter&                   operator=(const Interpreter&) = delete;
    std::unique_ptr<MemoryManager> MemManager;
};

/// Helper class to push and pop a function frame in a C++ scope.
class ScopedFunctionFrame {
  private:
    Interpreter& interpreter;

  public:
    explicit ScopedFunctionFrame(Interpreter& interpreter) : interpreter(interpreter) { interpreter.pushFunctionFrame(); }

    ~ScopedFunctionFrame() { interpreter.popFunctionFrame(); }
};

/// Helper class to push and pop a region frame in a C++ scope.
class ScopedRegionFrame {
  private:
    Interpreter& interpreter;

  public:
    explicit ScopedRegionFrame(Interpreter& interpreter) : interpreter(interpreter) { interpreter.pushRegionFrame(); }

    ~ScopedRegionFrame() { interpreter.popRegionFrame(); }
};

}  // namespace mlir

#endif  // MLIR_INTERPRETER_INTERPRETER_H_
