//===- MemoryManager.h
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
#ifndef MLIR_INTERPRETER_MEMORYMANAGER_H
#define MLIR_INTERPRETER_MEMORYMANAGER_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "MLIFormat.h"

namespace mlir {

class MemoryManager {
  public:
    // Use a "virtual address" for reading/writing into the buffer
    // The upper 32 bits give a unique allocation ID, which is assigned during allocate()
    // Note that over the course of execution, the associated base address for a given allocation may be reassigned
    // This is why we use this virtual scheme, since we can't update an address in the interpreter w/o violating SSA
    // The lower 32 bits is an offset from the allocation's base address from which to start performing memory operations
    using VirtualAddr = uint64_t;

  public:
    virtual ~MemoryManager()                                                         = default;
    virtual VirtualAddr allocate(size_t size)                                        = 0;
    virtual void        free(VirtualAddr addr)                                       = 0;
    virtual void        read(VirtualAddr addr, void* dst, size_t size) const         = 0;
    virtual void        write(VirtualAddr addr, const void* src, size_t size)        = 0;
    virtual void        copy(const VirtualAddr src, const uint64_t dst, size_t size) = 0;
    virtual bool        realloc(const VirtualAddr src, const size_t new_size)        = 0;
    virtual void        force_write(VirtualAddr addr, const void* src, size_t size)  = 0;

  public:
    static inline VirtualAddr make_vaddr(uint32_t alloc_id, uint32_t offset) { return uint64_t(alloc_id) << 32 | offset; }

    static inline uint32_t get_vaddr_id(VirtualAddr vaddr) { return vaddr >> 32; }

    static inline uint32_t get_vaddr_offset(VirtualAddr vaddr) { return vaddr & 0xFFFFFFFF; }
};

/// TODO: Move to another file maybe?
/// A naive memory manager that stores all allocations in one contiguous buffer.
/// For each allocation, it:
///  - Returns a 64-bit "address" (offset into the buffer).
///  - Performs simple bounds checking on read/write.
class SimpleMemoryManager : public MemoryManager {
  public:
    explicit SimpleMemoryManager(size_t initialSize = 1024 * 1024) {
        // Pre-allocate our contiguous memory space
        Mem.resize(initialSize, 0);
        freeBlocks.push_back({0, initialSize});
        next = freeBlocks.begin();
    }

    ~SimpleMemoryManager() {
        if ( allocations.empty() )
            return;
        llvm::errs() << mli::fmt::warning("Failed to free allocations\n");
        for ( const auto& [id, alloc] : allocations ) {
            llvm::errs() << " - address " << id << " with size " << alloc.size << "\n";
        }
    }

    /// Allocate 'size' bytes. We return an offset (uint64_t) into our single buffer.
    /// Throws std::bad_alloc if we can't fit the allocation.
    VirtualAddr allocate(size_t size) override {
        FreeBlock& new_block = findAvailableBlock(size);
        Allocation alloc     = {size, new_block.base_addr, next_alloc_id};

        // Adjust free block dimensions
        new_block.size -= size;
        new_block.base_addr += size;

        allocations[next_alloc_id++] = alloc;
        return make_vaddr(alloc.alloc_id, 0);
    }

    /// Remove allocation associated with addr from map
    /// Also reclaims freed memory for future allocations
    void free(VirtualAddr vaddr) override {
        uint32_t alloc_id     = get_vaddr_id(vaddr);
        // NOTE: We ignore the offset when freeing memory, instead clamping it to zero
        Allocation& alloc     = findAllocation(alloc_id);
        size_t      freedSize = alloc.size;
        uint32_t    addr      = alloc.base_addr;
        allocations.erase(alloc_id);

        // Consolidate blocks by reclaiming freed memory
        // Find supremum/infimum of freed block located at addr
        // (i.e. closest available blocks on either side of freed block)
        auto right     = std::upper_bound(freeBlocks.begin(), freeBlocks.end(), addr, [](const uint32_t a, const FreeBlock& b) {
            return a <= b.base_addr;
        });
        auto left      = std::prev(right);

        // Test if the adjacent blocks of the freed block are available
        // This determines how the blocks can be consolidated
        bool openLeft  = left->base_addr + left->size == addr;
        bool openRight = right->base_addr == addr + freedSize;

        if ( openLeft && openRight ) {
            // Case I: Incorporate both the freed block and right into left
            left->size += (freedSize + right->size);
            // The right pointer is now redundant, so we can delete it
            // Ensure that next != right to prevent dangling pointer
            if ( next == right ) {
                next = ++next == freeBlocks.end() ? freeBlocks.begin() : next;
            }
            freeBlocks.erase(right);  // made redundant by expansion
        }
        else if ( openLeft && !openRight ) {
            // Case II: Incorporate the freed block into left
            left->size += freedSize;
        }
        else if ( !openLeft && openRight ) {
            // Case III: Incorporate the freed block into right
            right->base_addr = addr;
            right->size += freedSize;
        }
        else {
            // Case IV: Bookended by allocated memory, create new entry between left and right
            FreeBlock newBlock = {addr, freedSize};
            freeBlocks.insert(right, newBlock);
        }
    }

    /// Read 'size' bytes from address 'addr' into 'dst'.
    /// Performs simple bounds checking against our recorded allocations.
    void read(VirtualAddr vaddr, void* dst, size_t size) const override {
        const Allocation& alloc = findAllocation(get_vaddr_id(vaddr));
        // Ensure the entire read fits within [allocStart, allocStart + allocSize)
        if ( size > alloc.size ) {
            throw std::runtime_error("SimpleMemoryManager: read out of bounds");
        }
        // Do address translation and read from buffer
        std::memcpy(dst, &Mem[alloc.base_addr + get_vaddr_offset(vaddr)], size);
    }

    /// Write 'size' bytes from 'src' into address 'addr'.
    /// Performs simple bounds checking.
    void write(VirtualAddr vaddr, const void* src, size_t size) override {
        Allocation& alloc = findAllocation(get_vaddr_id(vaddr));
        // Ensure the entire write fits within [allocStart, allocStart + allocSize)
        if ( size > alloc.size ) {
            throw std::runtime_error("SimpleMemoryManager: write out of bounds");
        }
        std::memcpy(&Mem[alloc.base_addr + get_vaddr_offset(vaddr)], src, size);
    }

    /// Write 'size' bytes from 'src' into address 'addr', calling realloc if allocation is too small
    /// Throws exception iff realloc fails to allocate another block
    void force_write(VirtualAddr vaddr, const void* src, size_t size) override {
        Allocation& alloc = findAllocation(get_vaddr_id(vaddr));
        if ( size > alloc.size ) {
            realloc(vaddr, size);
        }
        std::memcpy(&Mem[alloc.base_addr + get_vaddr_offset(vaddr)], src, size);
    }

    /// Copy 'size' bytes from address 'src' to address 'dst'
    /// Performs simple bounds checking
    void copy(const VirtualAddr src, const VirtualAddr dst, size_t size) override {
        Allocation& src_alloc = findAllocation(get_vaddr_id(src));
        Allocation& dst_alloc = findAllocation(get_vaddr_id(dst));
        // Ensure the entire read fits within [allocStart, allocStart + allocSize)
        if ( size > src_alloc.size ) {
            throw std::runtime_error("SimpleMemoryManager: read out of bounds in copy");
        }
        if ( size > dst_alloc.size ) {
            throw std::runtime_error("SimpleMemoryManager: write out of bounds in copy");
        }
        std::memcpy(&Mem[dst_alloc.base_addr + get_vaddr_offset(dst)], &Mem[src_alloc.base_addr + get_vaddr_offset(src)], size);
    }

    /// Resize 'src' to 'new_size' bytes
    /// Return true iff base address changes
    bool realloc(const VirtualAddr vaddr, const size_t new_size) override {
        Allocation& src_alloc = findAllocation(get_vaddr_id(vaddr));

        // Requested size is smaller, no need to move
        if ( new_size < src_alloc.size ) {
            src_alloc.size = new_size;
            return false;
        }

        // New size is too large, reallocate elsewhere
        // Find free block immediately after allocation
        auto it = std::upper_bound(
            freeBlocks.begin(),
            freeBlocks.end(),
            src_alloc.base_addr + src_alloc.size,
            [](const size_t addr, const FreeBlock& blk) { return blk.base_addr >= addr; }
        );

        // Right neighbor is available, poach memory from it and keep our original address
        if ( it != freeBlocks.end() && it->base_addr == src_alloc.base_addr + src_alloc.size ) {
            // How much memory do we need to borrow?
            size_t delta = new_size - src_alloc.size;
            if ( it->size >= delta ) {
                it->size -= delta;
                it->base_addr += delta;
                src_alloc.size = new_size;
                return false;
            }
        }

        // Right neighbor is unavailable or inadequate, must relocate buffer
        free(vaddr);
        FreeBlock& new_block = findAvailableBlock(new_size);

        // Update dimensions of allocation
        src_alloc.base_addr  = new_block.base_addr;
        src_alloc.size       = new_size;

        // Update dimensions of free block
        new_block.base_addr += new_size;
        new_block.size -= new_size;
        return true;
    }

  private:
    /// A struct to track basic allocation metadata.
    /// TODO: Add alignment, original request, etc.
    struct Allocation {
        size_t   size;
        uint32_t base_addr;
        uint32_t alloc_id;
    };

    struct FreeBlock {
        uint32_t base_addr;
        size_t   size;
    };

    /// Lookup which allocation covers a given address 'addr'.
    /// We do a lower_bound or predecessor search to find the earliest allocation
    /// that starts before (or at) 'addr'.
    inline Allocation& findAllocation(uint32_t id) {
        auto it = allocations.find(id);
        if ( it == allocations.end() ) {
            throw std::runtime_error("ID not found in allocation");
        }
        return it->second;
    }

    inline const Allocation& findAllocation(uint32_t id) const {
        auto it = allocations.find(id);
        if ( it == allocations.end() ) {
            throw std::runtime_error("ID not found in allocation");
        }
        return it->second;
    }

    FreeBlock& findAvailableBlock(size_t size) {
        auto oldNext = next;
        do {
            if ( next->size >= size ) {
                return *next;
            }
            // Advance list, circling back if at end
            next = ++next == freeBlocks.end() ? freeBlocks.begin() : next;
        } while ( next != oldNext );

        // Examined all blocks, but none are large enough
        throw std::bad_alloc();
    }

    // A single contiguous buffer
    std::vector<char> Mem;

    // All available memory blocks
    std::list<FreeBlock> freeBlocks;

    // Iterator to the last allocated block in memory
    // Used for "next-block" allocation strategy
    std::list<FreeBlock>::iterator next;

    // Map from allocation ID -> Allocation metadata
    std::map<uint32_t, Allocation> allocations;

    // Allocation ID
    uint32_t next_alloc_id = 0;
};
}  // namespace mlir

// TODO: Eventaully add something for, say, a DRAMSimMemoryManager or other popular memory systems.

#endif  // MLIR_INTERPRETER_MEMORYMANAGER_H
