//===- MemRefInterpreter.cpp - MemRef dialect interpreter -------------*- C++ -*-===//
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

#include "ADT/MultiArray.h"
#include "MLIFormat.h"
#include "MLIUtils.h"
#include "mlir/Interpreter/Dialects/MemRefInterpreter.h"
#include "mlir/Interpreter/Interpreter.h"
#include "mlir/Interpreter/InterpreterOpInterface.h"

using namespace mlir;
using namespace mli;

namespace {

struct AllocOpInterpreter : public InterpreterOpInterface::ExternalModel<AllocOpInterpreter, memref::AllocOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        // NOTE: I'm assuming all operands correspond to the dimensions of the memref
        llvm::SmallVector<int64_t, 4> dims;
        for ( const EvalValue& val : operands ) {
            dims.push_back(val.getData<int64_t>().front());
        }

        // We need to figure out the element type for the memref, since MultiArray is templated
        auto memref_type        = mlir::dyn_cast<mlir::BaseMemRefType>(*(op->result_type_begin()));
        auto elem_type          = memref_type.getElementType();

        // Create memref and write it to our memory manager, which is placed on the heap
        MultiArray        buff  = MultiArray(elem_type, dims);
        std::vector<char> bytes = buff.serialize();
        uint64_t          addr  = interpreter.allocateInMemManager(bytes.size());
        interpreter.writeToMemManager(addr, bytes.data(), bytes.size());

        // If write is successful, bind the address to the SSA variable
        auto evalResult = interpreter.createEvalValue(LLVM::LLVMPointerType::get(interpreter.getContext()), &addr, sizeof(addr));
        return interpreter.createBindValueResult(evalResult);
    }
};

struct AllocaOpInterpreter : public InterpreterOpInterface::ExternalModel<AllocaOpInterpreter, memref::AllocaOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {

        // We need to figure out the element type for the memref, since MultiArray is templated
        auto memref_type             = mlir::dyn_cast<mlir::BaseMemRefType>(*(op->result_type_begin()));
        auto elem_type               = memref_type.getElementType();

        // This is a stack variable, so allocate the SSA variable as normal
        MultiArray        buff       = MultiArray(elem_type, memref_type.getShape());
        std::vector<char> bytes      = buff.serialize();
        auto              evalResult = interpreter.createEvalValue(memref_type, bytes.data(), bytes.size());

        return interpreter.createBindValueResult(evalResult);
    }
};

struct DeallocOpInterpreter : public InterpreterOpInterface::ExternalModel<DeallocOpInterpreter, memref::DeallocOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        // NOTE: This can ONLY be called on memrefs allocated by alloc
        // Lifetime of memrefs created via alloca are managed by the stack
        // As of now, any heap-allocated memref are stored in the interpreter's memory manager
        // The actual SSA variable which we will retrieve is an address (uint64_t), not a MultiArray type
        try {
            uint64_t addr_to_free = operands[0].getData<uint64_t>().front();
            interpreter.freeInMemManager(addr_to_free);
        } catch ( std::runtime_error& e ) {
            // We want a more informative error message that displays the SSA variable we failed to free
            // There's no native support for this, so parse the operation string to get it
            std::string              full_op_name;
            llvm::raw_string_ostream ss(full_op_name);
            op->print(ss);
            size_t start_idx = full_op_name.find_first_of('%');
            for ( size_t i = start_idx; i < full_op_name.length(); i++ ) {
                if ( std::isspace(full_op_name[i]) || full_op_name[i] == ':' ) {
                    std::string msg = "Failed to free SSA variable " + full_op_name.substr(start_idx, i - start_idx) +
                                      ", are you sure it was created with memref.alloc?";
                    return interpreter.createErrorResult(msg);
                }
            }
        }
        return interpreter.createVoidResult();
    }
};

struct StoreOpInterpreter : public InterpreterOpInterface::ExternalModel<StoreOpInterpreter, memref::StoreOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        // operands = [val_to_store, memref, idx1, idx2 ...]
        // Get indices from remaining operands
        llvm::SmallVector<intptr_t> indices;
        for ( int i = 2; i < operands.size(); i++ ) {
            intptr_t dim = operands[i].getData<intptr_t>().front();
            indices.push_back(dim);
        }

        const mlir::Type elem_type   = operands[0].getType();
        const auto       memref_type = mlir::dyn_cast<mlir::BaseMemRefType>(operands[1].getType());

        EvalValue    memref          = operands[1];
        const size_t total_bytes     = memref.getRawDataSizeInBytes();
        MultiArray   buff;
        if ( isa<LLVM::LLVMPointerType>(memref.getType()) ) {
            llvm::outs() << "Restoring from MemManager\n";
            char*    bytes = new char(total_bytes);
            uint64_t addr  = memref.getData<uint64_t>().front();
            interpreter.readFromMemManager(addr, bytes, total_bytes);
            buff.deserialize(bytes, total_bytes);
            delete bytes;
        }
        else {
            llvm::outs() << "Restoring from stack\n";
            // FIXME: This probably needs to be a MultiArray, not MultiArray*
            const char* bytes = memref.getRawData();
            buff.deserialize(bytes, total_bytes);
        }

        // Store the value
        // NOTE: Probably want to change this to a visitor pattern, if we ever extend types
        if ( isa<IndexType>(elem_type) ) {
            intptr_t val               = operands[0].getData<intptr_t>().front();
            buff.at<intptr_t>(indices) = val;
        }
        else if ( isa<FloatType>(elem_type) ) {
            llvm::APFloat val               = operands[0].getData<llvm::APFloat>().front();
            buff.at<llvm::APFloat>(indices) = val;
        }
        else if ( elem_type.isInteger(1) ) {
            bool val               = operands[0].getData<bool>().front();
            buff.at<char>(indices) = val;
        }
        else if ( isa<IntegerType>(elem_type) ) {
            llvm::APInt val               = operands[0].getData<llvm::APInt>().front();
            buff.at<llvm::APInt>(indices) = val;
        }

        buff.print();

        return interpreter.createVoidResult();
    }
};

}  // end anonymous namespace

void MemRefInterpreter::attachInterface(MLIRContext& context) {
    memref::AllocOp::attachInterface<AllocOpInterpreter>(context);
    memref::AllocaOp::attachInterface<AllocaOpInterpreter>(context);
    memref::DeallocOp::attachInterface<DeallocOpInterpreter>(context);
    memref::StoreOp::attachInterface<StoreOpInterpreter>(context);
}
