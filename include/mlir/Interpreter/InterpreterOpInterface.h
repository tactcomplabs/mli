//===- CopyOpInterface.h - copy operations interface ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for the interpretable
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INTERPRETEROPINTERFACE_H_
#define MLIR_INTERFACES_INTERPRETEROPINTERFACE_H_

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <utility>

namespace mlir {

class Interpreter;

namespace detail {
class EvalValueImpl : public llvm::RefCountedBase<EvalValueImpl> {
public:
  constexpr static size_t kAlign = 8;

public:
  EvalValueImpl(Type type, size_t dataSize) : type(type), data(dataSize, 0) {}

  EvalValueImpl(Type type, llvm::ArrayRef<char> data)
      : type(type), data(data.begin(), data.end()) {}

  Type getType() const { return type; }

  size_t getRawDataSizeInBytes() const { return data.size(); }

  char *getRawData() { return data.data(); }

  const char *getRawData() const { return data.data(); }

private:
  Type type;
  std::vector<char> data;
};
} // namespace detail

/// A handle to the evaluated value.
class EvalValue {
private:
  friend class Interpreter;

public:
  EvalValue();
  ~EvalValue();
  EvalValue(const EvalValue &src);
  EvalValue &operator=(const EvalValue &src);
  EvalValue(EvalValue &&src);
  EvalValue &operator=(EvalValue &&src);

private:
  explicit EvalValue(detail::EvalValueImpl *impl);

public:
  /// Returns the type of the evaluated value.
  Type getType() const;

  /// Returns the data buffer size for the evaluated value.
  size_t getRawDataSizeInBytes() const;

  /// Returns the data buffer for the evaluated value.
  char *getRawData();

  /// Returns the data buffer for the evaluated value.
  const char *getRawData() const;

  template <typename DataType>
  llvm::MutableArrayRef<DataType> getData() {
    size_t dataSizeInBytes = getRawDataSizeInBytes();
    assert(dataSizeInBytes % sizeof(DataType) == 0);
    return llvm::MutableArrayRef<DataType>(
        reinterpret_cast<DataType *>(getRawData()),
        dataSizeInBytes / sizeof(DataType));
  }

  template <typename DataType>
  llvm::ArrayRef<DataType> getData() const {
    size_t dataSizeInBytes = getRawDataSizeInBytes();
    assert(dataSizeInBytes % sizeof(DataType) == 0);
    return llvm::ArrayRef<DataType>(
        reinterpret_cast<const DataType *>(getRawData()),
        dataSizeInBytes / sizeof(DataType));
  }

  /// Returns whether this handle holds a null value.
  explicit operator bool() const { return static_cast<bool>(impl); }

  /// Checks whether `this` and `rhs` refer to the same value.
  bool operator==(const EvalValue &rhs) const { return impl == rhs.impl; }

  /// Checks whether `this` and `rhs` refer to different values.
  bool operator!=(const EvalValue &rhs) const { return impl != rhs.impl; }

private:
  llvm::IntrusiveRefCntPtr<detail::EvalValueImpl> impl;
};

/// A class to represent interpreter evaluation errors.
class EvalError {
public:
  explicit EvalError(llvm::StringRef message,
                     bool shouldPrintStackTrace = false);

  /// Get the error message.
  const std::string &getMessage() const { return message; }

  /// Get the stack trace.
  const std::string &getStackTrace() const { return stacktrace; }

private:
  std::string message;
  std::string stacktrace;
};

/// Evaluation result states.
enum class EvalResultKind {
  /// An error occurs in the interpreter. In this state, `getError()` returns
  /// an `EvalError` object.
  Error,

  /// Bind evaluated values to the op result SSA names. In this state,
  /// `getValues()` returns the evaluated values which should be bound to SSA
  /// names.
  BindValue,

  /// Return evaluated values to the caller. In this state, `getValues()`
  /// returns the evaluated values which should be treated as the function
  /// return values.
  ///
  /// Note: Typically, `ReturnValue` may pop multiple regions until it is
  /// handled by a call op.
  ReturnValue,

  /// Return evaluated values to the parent op. In this state, `getValues()`
  /// returns the evaluated values which should be treated as the region yield
  /// values.
  ///
  /// Note: Typically, `YieldValue` only pops one region and will be handled by
  /// the parent op.
  YieldValue,

  /// Branch to a new block. In this state, `getBlock()` returns the
  /// destination block and `getValues()` returns the block arguments.
  Branch,
};

/// A class to represent the interpreter evaluation results.
class EvalResult {
  friend class Interpreter;

public:
  EvalResult() : kind(EvalResultKind::Error) {}

private:
  EvalResult(EvalResultKind kind, llvm::ArrayRef<EvalValue> values,
             Block *block, std::unique_ptr<EvalError> error)
      : kind(kind), values(values.begin(), values.end()), block(block),
        error(std::move(error)) {}

public:
  /// Get the evaluation result kind.
  EvalResultKind getKind() const { return kind; }

  /// Get the evaluated values for BindValue, ReturnValue, and YieldValue and
  /// the block arguments for Branch.
  llvm::ArrayRef<EvalValue> getValues() { return llvm::ArrayRef<EvalValue>(values); }
  llvm::ArrayRef<EvalValue> getValues() const {
    return llvm::ArrayRef<EvalValue>(values);
  }

  /// Get the branch destination block.
  Block *getBlock() const { return block; }

  /// Get the evaluation error.
  const EvalError &getError() const { return *error; }

private:
  EvalResultKind kind;
  llvm::SmallVector<EvalValue, 1> values;
  Block *block;
  std::unique_ptr<EvalError> error;
};

} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interpreter/InterpreterOpInterface.h.inc"

#endif // MLIR_INTERFACES_INTERPRETEROPINTERFACE_H_
