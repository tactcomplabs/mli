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

// The MultiArray class isn't trivially copyable, so we can't store it in the interpreter
// Instead, we write the raw bytes into the interpreter's memory manager
// The EvalValues associated with memrefs instead store the virtual address and size of the raw bytes
struct MemRefAllocation {
    uint64_t vaddr;
    size_t   size;

    MemRefAllocation(uint64_t a, size_t s) : vaddr(a), size(s) {};
};
using MemRefAllocation = struct MemRefAllocation;

// Create an instance of the MultiArray class from the MemoryManager
MultiArray restoreFromMemManager(const EvalValue& val, const Interpreter& interpreter) {
    auto [vaddr, total_bytes] = val.getData<MemRefAllocation>().front();
    std::vector<char> buff(total_bytes);
    interpreter.readFromMemManager(vaddr, buff.data(), total_bytes);
    MultiArray arr = MultiArray(buff.data(), total_bytes, vaddr);
    return arr;
}

// Update the byte representation of a MultiArray already present in the MemoryManager
size_t updateInMemManager(const MultiArray& arr, const Interpreter& interpreter) {
    const uint64_t          vaddr = arr.getAddr();
    const std::vector<char> bytes = arr.serialize();
    interpreter.forceWriteToMemManager(vaddr, bytes.data(), bytes.size());
    return bytes.size();
}

// Return the SSA name for a given variable for clearer error messages
std::string getSSAName(Operation* op) {
    // The EvalValue class doesn't track SSA names so parse the operation string to get it
    std::string              full_op_name;
    llvm::raw_string_ostream ss(full_op_name);
    op->print(ss);
    size_t start_idx = full_op_name.find_first_of('%');
    for ( size_t i = 1 + start_idx; i < full_op_name.length(); i++ ) {
        if ( !std::isalnum(full_op_name[i]) ) {
            return full_op_name.substr(start_idx, i - start_idx);
        }
    }
    return full_op_name;  // shouldn't happen
}

struct AllocOpInterpreter : public InterpreterOpInterface::ExternalModel<AllocOpInterpreter, memref::AllocOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        // We need to figure out the element type for the memref, since MultiArray is templated
        auto memref_type = mlir::dyn_cast<mlir::BaseMemRefType>(*(op->result_type_begin()));
        auto elem_type   = memref_type.getElementType();

        // Determine the dimensions of the memref
        // The getShape() function returns a vector of the correct rank and all dimensions known at compile-time
        // If a dimension isn't known until runtime, it must be supplied in operands as an EvalValue
        llvm::SmallVector<intptr_t> dims;
        llvm::ArrayRef<int64_t>     shape  = memref_type.getShape();
        size_t                      op_idx = 0;
        for ( int64_t val : shape ) {
            if ( val >= 0 ) {
                dims.push_back(val);
            }
            else {
                if ( op_idx >= operands.size() ) {
                    return interpreter.createErrorResult("Not enough arguments for dynamic shape\n");
                }
                dims.push_back(operands[op_idx].getData<intptr_t>().front());
                op_idx++;
            }
        }

        // Create memref and write it to our memory manager, which is placed on the heap
        MultiArray        buff  = MultiArray(elem_type, dims);
        std::vector<char> bytes = buff.serialize();
        uint64_t          vaddr = interpreter.allocateInMemManager(bytes.size());
        interpreter.writeToMemManager(vaddr, bytes.data(), bytes.size());

        // If write is successful, bind the virtual address to the SSA variable
        MemRefAllocation alloc      = MemRefAllocation(vaddr, bytes.size());
        auto             evalResult = interpreter.createEvalValue(memref_type, &alloc, sizeof(alloc));
        return interpreter.createBindValueResult(evalResult);
    }
};

struct AllocaOpInterpreter : public InterpreterOpInterface::ExternalModel<AllocaOpInterpreter, memref::AllocaOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        // We need to figure out the element type for the memref, since MultiArray is templated
        auto memref_type = mlir::dyn_cast<mlir::BaseMemRefType>(*(op->result_type_begin()));
        auto elem_type   = memref_type.getElementType();

        // Determine the dimensions of the memref
        // The getShape() function returns a vector of the correct rank and all dimensions known at compile-time
        // If a dimension isn't known until runtime, it must be supplied in operands as an EvalValue
        llvm::SmallVector<intptr_t> dims;
        llvm::ArrayRef<int64_t>     shape  = memref_type.getShape();
        size_t                      op_idx = 0;
        for ( int64_t val : shape ) {
            if ( val >= 0 ) {
                dims.push_back(val);
            }
            else {
                if ( op_idx >= operands.size() ) {
                    return interpreter.createErrorResult("Not enough arguments for dynamic shape\n");
                }
                dims.push_back(operands[op_idx].getData<intptr_t>().front());
                op_idx++;
            }
        }

        // Create memref and write it to our memory manager, which is placed on the heap
        MultiArray        buff  = MultiArray(elem_type, dims);
        std::vector<char> bytes = buff.serialize();
        uint64_t          vaddr = interpreter.allocateInMemManager(bytes.size());
        interpreter.writeToMemManager(vaddr, bytes.data(), bytes.size());

        // If write is successful, bind the virtual address to the SSA variable
        MemRefAllocation alloc      = MemRefAllocation(vaddr, bytes.size());
        auto             evalResult = interpreter.createEvalValue(memref_type, &alloc, sizeof(alloc));
        return interpreter.createBindValueResult(evalResult);
    }
};

struct CopyOpInterpreter : public InterpreterOpInterface::ExternalModel<CopyOpInterpreter, memref::CopyOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        MemRefAllocation src = operands[0].getData<MemRefAllocation>().front();
        MemRefAllocation dst = operands[1].getData<MemRefAllocation>().front();
        if ( src.size != dst.size ) {
            return interpreter.createErrorResult("Source and destination memrefs have different shapes");
        }
        interpreter.copyInMemManager(src.vaddr, dst.vaddr, src.size);
        return interpreter.createVoidResult();
    }
};

struct DeallocOpInterpreter : public InterpreterOpInterface::ExternalModel<DeallocOpInterpreter, memref::DeallocOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        // NOTE: This can ONLY be called on memrefs allocated by alloc
        // Lifetime of memrefs created via alloca are managed by the stack
        // As of now, any heap-allocated memref are stored in the interpreter's memory manager
        // The actual SSA variable which we will retrieve is an virtual address (uint64_t), not a MultiArray type
        try {
            auto [vaddr_to_free, size] = operands[0].getData<MemRefAllocation>().front();
            interpreter.freeInMemManager(vaddr_to_free);
            llvm::outs() << mli::fmt::info(
                "Freed memref allocation " + std::to_string(MemoryManager::get_vaddr_id(vaddr_to_free)) + "\n"
            );
        } catch ( std::runtime_error& e ) {
            return interpreter.createErrorResult(
                "Failed to free SSA variable " + getSSAName(op) + ", are you sure it was created with memref.alloc?"
            );
        }
        return interpreter.createVoidResult();
    }
};

struct DimOpInterpreter : public InterpreterOpInterface::ExternalModel<DimOpInterpreter, memref::DimOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        MultiArray arr     = restoreFromMemManager(operands[0], interpreter);
        intptr_t   dim_idx = operands[1].getData<intptr_t>().front();
        intptr_t   dim     = arr.dim(dim_idx);
        EvalValue  val     = interpreter.createEvalValue(mlir::IndexType::get(interpreter.getContext()), &dim, sizeof(dim));
        return interpreter.createBindValueResult(val);
    }
};

struct LoadOpInterpreter : public InterpreterOpInterface::ExternalModel<LoadOpInterpreter, memref::LoadOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        // operands = [val_to_store, memref, idx1, idx2 ...]
        // Get indices from remaining operands
        llvm::SmallVector<intptr_t> indices;
        for ( int i = 1; i < operands.size(); i++ ) {
            intptr_t dim = operands[i].getData<intptr_t>().front();
            indices.push_back(dim);
        }

        EvalValue        memref      = operands[0];
        const auto       memref_type = mlir::dyn_cast<mlir::BaseMemRefType>(memref.getType());
        const mlir::Type elem_type   = memref_type.getElementType();
        MultiArray       buff        = restoreFromMemManager(memref, interpreter);

        // Load the value
        // NOTE: Probably want to change this to a visitor pattern, if we ever extend types
        EvalValue res;
        try {
            if ( isa<IndexType>(elem_type) ) {
                intptr_t val = buff.at<intptr_t>(indices);
                res          = interpreter.createEvalValue(elem_type, &val, sizeof(val));
            }
            else if ( isa<FloatType>(elem_type) ) {
                llvm::APFloat val = buff.at<llvm::APFloat>(indices);
                res               = interpreter.createEvalValue(elem_type, &val, sizeof(val));
            }
            else if ( elem_type.isInteger(1) ) {
                bool val = buff.at<char>(indices);
                res      = interpreter.createEvalValue(elem_type, &val, sizeof(val));
            }
            else if ( isa<IntegerType>(elem_type) ) {
                llvm::APInt val = buff.at<llvm::APInt>(indices);
                res             = interpreter.createEvalValue(elem_type, &val, sizeof(val));
            }
        } catch ( std::exception& e ) {
            std::string msg = e.what();
            return interpreter.createErrorResult(msg + " for memref " + getSSAName(op));
        }

        return interpreter.createBindValueResult(res);
    }
};

struct ReshapeOpInterpreter : public InterpreterOpInterface::ExternalModel<ReshapeOpInterpreter, memref::ReshapeOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        MultiArray    buff      = restoreFromMemManager(operands[0], interpreter);
        MultiArray    dims_buff = restoreFromMemManager(operands[1], interpreter);
        const int64_t num_elems = dims_buff.getNumElements();

        // The documentation mentions that the shape memref can be of either integer of index type
        // However, mlir-opt complains if you attempt to use integers as indices, so we enforce the index type
        if ( dims_buff.getKind() != MultiArray::MultiArrayKind::IndexKind ) {
            return interpreter.createErrorResult("Shape memref must be of index type");
        }

        const intptr_t* new_dims = dims_buff.getData<const intptr_t>();
        buff.reshape(llvm::ArrayRef(new_dims, num_elems));
        llvm::outs() << mli::fmt::info("new dims: " + mli::printAsList(buff.getDims()) + "\n");

        size_t           size       = updateInMemManager(buff, interpreter);
        MemRefAllocation alloc      = MemRefAllocation(buff.getAddr(), size);
        auto             evalResult = interpreter.createEvalValue(op->getResult(0).getType(), &alloc, sizeof(alloc));
        return interpreter.createBindValueResult(evalResult);
    }
};

struct StoreOpInterpreter : public InterpreterOpInterface::ExternalModel<StoreOpInterpreter, memref::StoreOp> {
    static EvalResult interpret(Operation* op, Interpreter& interpreter, ArrayRef<EvalValue> operands) {
        // operands = [val_to_store, memref, idx1, idx2 ...
        const mlir::Type elem_type = operands[0].getType();
        EvalValue        memref    = operands[1];
        MultiArray       buff      = restoreFromMemManager(memref, interpreter);

        // Get indices from remaining operands
        llvm::SmallVector<intptr_t> indices;
        for ( int i = 2; i < operands.size(); i++ ) {
            intptr_t dim = operands[i].getData<intptr_t>().front();
            indices.push_back(dim);
        }

        // Store the value
        // NOTE: Probably want to change this to a visitor pattern, if we ever extend types
        try {
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
        } catch ( std::exception& e ) {
            std::string msg = e.what();
            return interpreter.createErrorResult(msg + " for memref " + getSSAName(op));
        }

        llvm::outs() << mli::fmt::info("arr: " + buff.print() + "\n");
        updateInMemManager(buff, interpreter);

        return interpreter.createVoidResult();
    }
};

}  // end anonymous namespace

void MemRefInterpreter::attachInterface(MLIRContext& context) {
    memref::AllocOp::attachInterface<AllocOpInterpreter>(context);
    memref::AllocaOp::attachInterface<AllocaOpInterpreter>(context);
    memref::CopyOp::attachInterface<CopyOpInterpreter>(context);
    memref::DeallocOp::attachInterface<DeallocOpInterpreter>(context);
    memref::DimOp::attachInterface<DimOpInterpreter>(context);
    memref::LoadOp::attachInterface<LoadOpInterpreter>(context);
    memref::ReshapeOp::attachInterface<ReshapeOpInterpreter>(context);
    memref::StoreOp::attachInterface<StoreOpInterpreter>(context);
}
