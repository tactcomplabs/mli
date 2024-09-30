//===- InterpreterOpInterface.td - Interpreter interface ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an op interface for interpretable operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERPRETEROPINTERFACE

include "mlir/IR/OpBase.td"

def InterpreterOpInterface : OpInterface<"InterpreterOpInterface"> {
  let description = [{
    This is the OpInterface for the interpreter implementation.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns the results of the op given the input arguments.
      }],
      "::mlir::EvalResult", "interpret",
      (ins "::mlir::Interpreter &":$interpreter,
           "::llvm::ArrayRef<::mlir::EvalValue>":$operands)
    >,
  ];

  let cppNamespace = "::mlir";
}

#endif // MLIR_INTERPRETEROPINTERFACE
