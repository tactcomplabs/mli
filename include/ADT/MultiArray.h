/*
# ==========================- MultiArray.h -==========================
#
#
# Copyright (C) 2017-2025 Tactical Computing Laboratories, LLC
# All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include "MLIUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <memory>
#include <numeric>
#include <vector>

// MultiArray: A multidimensional array class
// MLIR supports vectors of arbitrary dimensions
// These can be standard arrays, tensors, or memrefs
namespace mli {

class MultiArray {
  public:
    // LLVM RTTI requires a class to provide a "kind" for distinguishing a subclass
    // Right now, the MultiArray class isn't extended from some base virtual class
    // However, this is more ergonomic than storing the mlir::Type, since we can just switch over an enum
    // This also hides some idiosyncracies related to type implementation (e.g bool = 1 bit integer)
    enum MultiArrayKind : uint8_t { BoolKind, IntegerKind, FloatKind, IndexKind, UnknownKind };

    static constexpr size_t kind_sizes[5] = {1, sizeof(llvm::APInt), sizeof(llvm::APFloat), sizeof(intptr_t), 1};

    // MultiArray isn't trivially copyable, so we can't directly attach it to an EvalValue
    // We opt to serialize into raw bytes as a workaround
    struct MultiArrayMetadata {
        int64_t        total_elems;
        MultiArrayKind elem_kind;
        uint64_t       buff_offset;  // start of data buffer
        uint64_t       buff_size;    // sizeof(buff)
        uint64_t       dims_offset;  // start of dims array
        uint64_t       dims_size;    // sizeof(dims)
    };

  private:
    std::vector<char>     buff;
    std::vector<intptr_t> dims;
    int64_t               total_elems;
    MultiArrayKind        elem_kind = UnknownKind;
    uint64_t              vaddr;  // virtual address in MemoryManager
  public:
    // Rule of 5
    MultiArray()                              = default;
    MultiArray& operator=(MultiArray& other)  = default;
    MultiArray(MultiArray&& other)            = default;
    MultiArray& operator=(MultiArray&& other) = default;
    ~MultiArray()                             = default;

    // Default constructor
    MultiArray(const mlir::Type t, llvm::ArrayRef<intptr_t> new_dims) {
        dims.assign(new_dims.begin(), new_dims.end());
        total_elems = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<intptr_t>());
        elem_kind   = getKindFromType(t);
        buff.resize(kind_sizes[elem_kind] * total_elems);
        if ( elem_kind == MultiArrayKind::FloatKind ) {
            llvm::APFloat* typed_arr = reinterpret_cast<llvm::APFloat*>(buff.data());
            llvm::APFloat  zero      = llvm::APFloat(llvm::APFloatBase::EnumToSemantics(mlir::EvalValue::getFloatSemantics(t)));
            std::uninitialized_fill_n(typed_arr, total_elems, zero);
        }
        else if ( elem_kind == MultiArrayKind::IntegerKind ) {
            llvm::APInt* typed_arr = reinterpret_cast<llvm::APInt*>(buff.data());
            std::uninitialized_fill_n(typed_arr, total_elems, llvm::APInt(t.getIntOrFloatBitWidth(), 0ull));
        }
        // All other types are POD, no need to call constructors
    }

    // Deserialization constructor, assuming that bytes was created with serialize
    MultiArray(const char* bytes, const size_t size, const uint64_t addr) : vaddr(addr) {
        assert(size >= sizeof(MultiArrayMetadata) && "bytes array is too small!");
        const auto* header     = reinterpret_cast<const MultiArrayMetadata*>(bytes);

        // Find where the data buffer is
        const char* buff_start = bytes + header->buff_offset;
        buff.assign(buff_start, buff_start + header->buff_size);

        // Find where the dimension vector is
        const intptr_t* dims_start = reinterpret_cast<const intptr_t*>(bytes + header->dims_offset);
        dims.assign(dims_start, dims_start + header->dims_size / sizeof(intptr_t));

        // Restore other primitives
        elem_kind   = header->elem_kind;
        total_elems = header->total_elems;
    }

    // Get non-const reference to a particular array element
    template<typename T>
    T& at(llvm::ArrayRef<intptr_t> indices) {
        if ( indices.size() != dims.size() ) {
            throw std::runtime_error("Incorrect number of indices supplied");
        }
        intptr_t idx    = 0;
        intptr_t stride = 1;
        assert(buff.size() % sizeof(T) == 0);
        T* typed_arr = reinterpret_cast<T*>(buff.data());
        for ( int i = dims.size() - 1; i >= 0; i-- ) {
            if ( indices[i] >= dims[i] ) {
                throw std::out_of_range("Index " + std::to_string(i) + " is out of bounds");
            }
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        return typed_arr[idx];
    }

    // Get const reference to a particular array element
    template<typename T>
    const T& at(llvm::ArrayRef<intptr_t> indices) const {
        if ( indices.size() != dims.size() ) {
            throw std::runtime_error("Incorrect number of indices supplied");
        }
        intptr_t idx    = 0;
        intptr_t stride = 1;
        assert(buff.size() % sizeof(T) == 0);
        const T* typed_arr = reinterpret_cast<const T*>(buff.data());
        for ( int i = dims.size() - 1; i >= 0; i-- ) {
            if ( indices[i] >= dims[i] ) {
                throw std::out_of_range("Index " + std::to_string(i) + " is out of bounds");
            }
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        return typed_arr[idx];
    }

    // Change the dimensions of the underlying array
    // The total number of elements must be conserved
    void reshape(llvm::ArrayRef<intptr_t> new_dims) {
        intptr_t new_total = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<intptr_t>());
        if ( new_total != total_elems ) {
            throw std::runtime_error("New shape has different number of elements compared to old shape");
        }
        dims.assign(new_dims.begin(), new_dims.end());
    }

    // Get dimension of specified axis
    intptr_t dim(const intptr_t dim_idx) const { return dims.at(dim_idx); }

    // Print the contents of the array
    std::string print() const {
        switch ( elem_kind ) {
            case BoolKind: return print_helper<char>(buff.data(), buff.size());
            case IntegerKind: return print_helper<llvm::APInt>(buff.data(), buff.size());
            case FloatKind: return print_helper<llvm::APFloat>(buff.data(), buff.size());
            case IndexKind: return print_helper<int64_t>(buff.data(), buff.size());
            default: llvm_unreachable("Unrecognized element type");
        }
    }

    llvm::ArrayRef<intptr_t> getDims() const { return llvm::ArrayRef(dims); }

    // Serialization logic for MemManager
    std::vector<char> serialize() const {
        // Create a buffer with the following contiguous structure:
        // [header_metadata, buff, dims]
        // where the header data tells us how to restore the buff and dims vectors
        MultiArrayMetadata header;
        header.total_elems = total_elems;
        header.elem_kind   = elem_kind;

        // Calculate offset for buff vector
        header.buff_offset = sizeof(header);
        header.buff_size   = buff.size();

        // Calculate offset for dims vector
        header.dims_offset = header.buff_offset + header.buff_size;
        header.dims_size   = dims.size() * sizeof(intptr_t);

        // Store everything into bytes vector
        std::vector<char> bytes(header.dims_offset + header.dims_size);
        std::memcpy(bytes.data(), &header, sizeof(header));
        std::memcpy(bytes.data() + header.buff_offset, buff.data(), header.buff_size);
        std::memcpy(bytes.data() + header.dims_offset, dims.data(), header.dims_size);
        return bytes;
    }

    // Get size of serialized MultiArray in bytes
    size_t getSerializedSize() const { return sizeof(MultiArrayMetadata) + buff.size() + dims.size() * sizeof(intptr_t); }

    template<typename T>
    const T* getData() const {
        assert(buff.size() % sizeof(T) == 0);
        return reinterpret_cast<const T*>(buff.data());
    }

    int64_t getNumElements() const { return total_elems; }

    MultiArrayKind getKind() const { return elem_kind; }

    void setAddr(const uint64_t newAddr) { vaddr = newAddr; }

    uint64_t getAddr() const { return vaddr; }

  private:
    template<typename T>
    std::string print_helper(const void* data, const size_t size) const {
        assert(size % sizeof(T) == 0);
        const T* typed_arr = reinterpret_cast<const T*>(data);

        if ( size == 0 )
            return "[]";
        std::string              out = "[";
        llvm::raw_string_ostream os(out);
        size_t                   num_elems = size / sizeof(T);

        for ( size_t i = 0; i < num_elems - 1; i++ ) {
// Older versions of LLVM do not overload the << operator for APFloat
// Detect this at compile-time and figure out how to print
#if NEW_LLVM
            os << typed_arr[i] << ", ";
#else
            if constexpr ( std::is_same_v<T, llvm::APFloat> ) {
                typed_arr[i].print(os);
                os << ", ";
            }
            else {
                os << typed_arr[i] << ", ";
            }
#endif
        }
#if NEW_LLVM
        os << typed_arr[num_elems - 1] << "]";
#else
        if constexpr ( std::is_same_v<T, llvm::APFloat> ) {
            typed_arr[num_elems - 1].print(os);
            os << "]";
        }
        else {
            os << typed_arr[num_elems - 1] << "]";
        }
#endif

        return os.str();
    }

    MultiArrayKind getKindFromType(const mlir::Type t) const {
        if ( t.isInteger(1) ) {  // 1 bit integer => bool
            return BoolKind;
        }
        if ( llvm::isa<mlir::IntegerType>(t) ) {
            return IntegerKind;
        }
        if ( llvm::isa<mlir::FloatType>(t) ) {
            return FloatKind;
        }
        if ( llvm::isa<mlir::IndexType>(t) ) {
            return IndexKind;
        }
        return UnknownKind;
    }
};

}  // namespace mli
