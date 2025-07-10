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

// LLVM requires some boilerplate for its RTTI
// Provide kinds for use in classof()
// NOTE: We must have a kind for every type T we expect in the array
// Right now, we just have bool, integer, float, and index
// If we want/need more, they'll have to be added as kinds
enum MultiArrayKind { BoolKind, IntegerKind, FloatKind, IndexKind };

class MultiArrayBase {
  protected:
    MultiArrayKind          kind;
    virtual MultiArrayBase* clone_impl() const = 0;

  public:
    virtual ~MultiArrayBase()                               = default;
    virtual void reshape(llvm::ArrayRef<intptr_t> new_dims) = 0;
    virtual void print() const                              = 0;

    MultiArrayKind getKind() const { return kind; }

    auto clone() const { return std::unique_ptr<MultiArrayBase>(clone_impl()); }
};

template<typename T>
class MultiArrayImpl : public MultiArrayBase {
  private:
    llvm::SmallVector<intptr_t> dims;
    std::vector<T>              arr;
    intptr_t                    total_elems;

  protected:
    virtual MultiArrayImpl* clone_impl() const override { return new MultiArrayImpl(*this); }

  public:
    ~MultiArrayImpl() override = default;

    // Default constructor for when T has a default constructor
    MultiArrayImpl(llvm::ArrayRef<intptr_t> init_dims) : dims(init_dims) {
        total_elems = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<intptr_t>());
        arr.reserve(total_elems);
        if constexpr ( std::is_same_v<T, bool> ) {
            kind = BoolKind;
        }
        kind = IndexKind;  // TODO: Is this a good idea for the default?
    };

    // Constructor for when T = APInt
    // This must be handled separately, since there's extra information that needs to be passed to the APInt ctr
    template<typename U = T, typename std::enable_if_t<std::is_same_v<U, llvm::APInt>, int> = 0>
    MultiArrayImpl(llvm::ArrayRef<intptr_t> init_dims, const unsigned width) : dims(init_dims) {
        total_elems = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<intptr_t>());
        arr.assign(total_elems, llvm::APInt(width, 0));
        kind = IntegerKind;
    }

    // Constructor for when T = APFloat
    // This must be handled separately, since there's extra information that needs to be passed to the APFloat ctr
    template<typename U = T, typename std::enable_if_t<std::is_same_v<U, llvm::APFloat>, int> = 0>
    MultiArrayImpl(llvm::ArrayRef<intptr_t> init_dims, const llvm::APFloatBase::Semantics& s) : dims(init_dims) {
        total_elems = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<intptr_t>());
        arr.assign(total_elems, llvm::APFloat(llvm::APFloatBase::EnumToSemantics(s)));
        kind = FloatKind;
    }

    T& at(llvm::ArrayRef<intptr_t> indices) {
        intptr_t idx    = 0;
        intptr_t stride = 1;
        assert(indices.size() == dims.size() && "Incorrect number of indices supplied");
        for ( int i = dims.size() - 1; i >= 0; i-- ) {
            assert(indices[i] < dims[i] && "Index out of range");
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        return arr.at(idx);
    }

    const T& at(llvm::ArrayRef<intptr_t> indices) const {
        intptr_t idx    = 0;
        intptr_t stride = 1;
        assert(indices.size() == dims.size() && "Incorrect number of indices supplied");
        for ( int i = dims.size() - 1; i >= 0; i-- ) {
            assert(indices[i] < dims[i] && "Index out of range");
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        return arr.at(idx);
    }

    void reshape(llvm::ArrayRef<intptr_t> new_dims) override {
        intptr_t new_total = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<intptr_t>());
        if ( new_total != total_elems ) {
            throw std::runtime_error("New shape has different number of elements compared to old shape");
        }
        dims.append(new_dims.begin(), new_dims.end());
    }

    void print() const override {
        if ( arr.empty() ) {
            llvm::outs() << "[]\n";
            return;
        }
        llvm::outs() << "[";
        for ( int i = 1; i < total_elems - 1; i++ ) {
            llvm::outs() << arr[i] << ", ";
        }
        llvm::outs() << arr[total_elems - 1] << "]\n";
    }

    static bool classof(const MultiArrayBase* base) {
        if constexpr ( std::is_same_v<T, bool> ) {
            return base->getKind() == BoolKind;
        }
        if constexpr ( std::is_same_v<T, intptr_t> ) {
            return base->getKind() == IndexKind;
        }
        if constexpr ( std::is_same_v<T, llvm::APInt> ) {
            return base->getKind() == IntegerKind;
        }
        if constexpr ( std::is_same_v<T, llvm::APFloat> ) {
            return base->getKind() == FloatKind;
        }
        return false;  // unrecognized type for T
    }
};

class MultiArray {
    std::unique_ptr<MultiArrayBase> impl;

  public:
    // Rule of 5
    MultiArray()  = default;

    ~MultiArray() = default;

    MultiArray(const MultiArray& other) : impl(other.impl->clone()) {}

    MultiArray& operator=(const MultiArray& other) {
        impl = other.impl->clone();
        return *this;
    }

    MultiArray(MultiArray&& other)            = default;
    MultiArray& operator=(MultiArray&& other) = default;

    // Custom constructor
    MultiArray(const mlir::Type elem_type, llvm::ArrayRef<intptr_t> dims) {
        if ( elem_type.isIndex() ) {
            impl = std::make_unique<MultiArrayImpl<intptr_t>>(dims);
        }
        else if ( elem_type.isInteger() ) {
            unsigned width = elem_type.getIntOrFloatBitWidth();
            impl           = std::make_unique<MultiArrayImpl<llvm::APInt>>(dims, width);
        }
        else if ( elem_type.isIntOrFloat() ) {  // is a float
            llvm::APFloat::Semantics s = getFloatSemantics(elem_type);
            impl                       = std::make_unique<MultiArrayImpl<llvm::APFloat>>(dims, s);
        }
    }

    template<typename T>
    T& at(llvm::ArrayRef<intptr_t> indices) {
        if ( auto ptr = llvm::dyn_cast<MultiArrayImpl<T>>(impl.get()) ) {
            return ptr->at(indices);
        }
        throw std::bad_cast();
    }

    template<typename T>
    const T& at(const llvm::ArrayRef<intptr_t> indices) const {
        if ( auto ptr = llvm::dyn_cast<MultiArrayImpl<T>>(impl.get()) ) {
            return ptr->at(indices);
        }
        throw std::bad_cast();
    }

    void reshape(llvm::ArrayRef<intptr_t> new_dims) { return impl->reshape(new_dims); }

    void print() const { return impl->print(); }
};

}  // namespace mli
