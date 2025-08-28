#ifndef MLIR_INTERPRETER_CUSTOMTYPES_H
#define MLIR_INTERPRETER_CUSTOMTYPES_H

#include <cstddef>
#include <cstdint>

namespace mlir {
using index_t = intptr_t;

// The MultiArray class isn't trivially copyable, so we can't store it in the interpreter
// Instead, we write the raw bytes into the interpreter's memory manager
// The EvalValues associated with memrefs instead store the virtual address and size of the raw bytes
struct MemRefAllocation {
    uint64_t vaddr;
    size_t   size;

    MemRefAllocation(uint64_t a, size_t s) : vaddr(a), size(s) {};
};
}  // namespace mlir
#endif
