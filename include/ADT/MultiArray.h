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

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

// MultiArray: A multidimensional array class
// MLIR supports vectors of arbitrary dimensions
// These can be standard arrays, tensors, or memrefs
namespace mli {

// LLVM RTTI requires a class to provide a "kind" for distinguishing a subclass
// Right now, the MultiArray class isn't extended from some base virtual class
// However, this is more ergonomic than storing the mlir::Type, since we can just switch over an enum
// This also hides some idiosyncracies related to type implementation (e.g bool = 1 bit integer, )
enum MultiArrayKind : uint8_t { BoolKind, IntegerKind, FloatKind, IndexKind, UnknownKind };

static constexpr size_t kind_sizes[5] = {1, sizeof(llvm::APInt), sizeof(llvm::APFloat), sizeof(intptr_t), 1};

class MultiArray {
    std::vector<char>    buff;
    std::vector<int64_t> dims;
    int64_t              total_elems;
    MultiArrayKind       elem_kind = UnknownKind;

  public:
    // Rule of 5
    MultiArray()                              = default;
    ~MultiArray()                             = default;
    MultiArray& operator=(MultiArray& other)  = default;
    MultiArray(MultiArray&& other)            = default;
    MultiArray& operator=(MultiArray&& other) = default;

    // Default constructor
    MultiArray(const mlir::Type t, llvm::ArrayRef<int64_t> new_dims) {
        dims.assign(new_dims.begin(), new_dims.end());
        total_elems = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<intptr_t>());
        elem_kind   = getKindFromType(t);
        buff.resize(kind_sizes[elem_kind] * total_elems);
        buff[0] = 'a';
    }

    // Restore from raw data
    void restore(const mlir::Type t, const char* raw_data, const size_t size_in_bytes, llvm::ArrayRef<int64_t> new_dims) {
        dims.assign(new_dims.begin(), new_dims.end());
        buff.assign(raw_data, raw_data + size_in_bytes);
        elem_kind = getKindFromType(t);
    }

    template<typename T>
    T& at(llvm::ArrayRef<intptr_t> indices) {
        intptr_t idx    = 0;
        intptr_t stride = 1;
        assert(buff.size() % sizeof(T) == 0);
        T* typed_arr = reinterpret_cast<T*>(buff.data());
        assert(indices.size() == dims.size() && "Incorrect number of indices supplied");
        for ( int i = dims.size() - 1; i >= 0; i-- ) {
            assert(indices[i] < dims[i] && "Index out of range");
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        return typed_arr[idx];
    }

    template<typename T>
    const T& at(const llvm::ArrayRef<intptr_t> indices) const {
        intptr_t idx    = 0;
        intptr_t stride = 1;
        assert(buff.size() % sizeof(T) == 0);
        const T* typed_arr = reinterpret_cast<const T*>(buff.data());
        assert(indices.size() == dims.size() && "Incorrect number of indices supplied");
        for ( int i = dims.size() - 1; i >= 0; i-- ) {
            assert(indices[i] < dims[i] && "Index out of range");
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        return typed_arr[idx];
    }

    void reshape(llvm::ArrayRef<int64_t> new_dims) {
        intptr_t new_total = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<intptr_t>());
        if ( new_total != total_elems ) {
            throw std::runtime_error("New shape has different number of elements compared to old shape");
        }
        dims.assign(new_dims.begin(), new_dims.end());
    }

    void print() const {
        if ( total_elems == 0 ) {
            llvm::outs() << "[]\n";
            return;
        }
        switch ( elem_kind ) {
            case BoolKind: print_helper<char>(); break;
            case IntegerKind: print_helper<llvm::APInt>(); break;
            case FloatKind: print_helper<llvm::APFloat>(); break;
            case IndexKind: print_helper<intptr_t>(); break;
            default: llvm_unreachable("Unrecognized element type");
        }
    }

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

    std::vector<char> serialize() const {
        // We'll create the buffer with the following contiguous structure:
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
        header.dims_size   = dims.size() * sizeof(int64_t);

        // Store everything into bytes vector
        std::vector<char> bytes(header.dims_offset + header.dims_size);
        std::memcpy(bytes.data(), &header, sizeof(header));
        std::memcpy(bytes.data() + header.buff_offset, buff.data(), header.buff_size);
        std::memcpy(bytes.data() + header.dims_offset, dims.data(), header.dims_size);
        return bytes;
    }

    // Deserialize from raw bytes
    // Requires bytes to be in the format specified by serialize()
    void deserialize(const char* bytes, const size_t size) {
        assert(size >= sizeof(MultiArrayMetadata) && "bytes array is too small!");
        const auto* header     = reinterpret_cast<const MultiArrayMetadata*>(bytes);

        // Find where the data buffer is
        const char* buff_start = bytes + header->buff_offset;
        buff.assign(buff_start, buff_start + header->buff_size);

        // Find where the dimension vector is
        const int64_t* dims_start = reinterpret_cast<const int64_t*>(bytes + header->dims_offset);
        dims.assign(dims_start, dims_start + header->dims_size / sizeof(int64_t));

        // Restore other primitives
        elem_kind   = header->elem_kind;
        total_elems = header->total_elems;
    }

  private:
    template<typename T>
    void print_helper() const {
        assert(buff.size() % sizeof(T) == 0);
        const T* typed_arr = reinterpret_cast<const T*>(buff.data());
        llvm::outs() << "[";
        for ( int i = 0; i < total_elems - 1; i++ ) {
            llvm::outs() << typed_arr[i] << ", ";
        }
        llvm::outs() << typed_arr[total_elems - 1] << "]\n";
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
