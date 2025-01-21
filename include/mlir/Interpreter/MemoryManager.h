#ifndef MLIR_INTERPRETER_MEMORYMANAGER_H
#define MLIR_INTERPRETER_MEMORYMANAGER_H

#include <algorithm>
#include <list>
#include <vector>
#include <map>

namespace mlir{

class MemoryManager {
public:
  virtual ~MemoryManager() = default;
  virtual uint64_t allocate(size_t size) = 0;
  virtual void free(uint64_t addr) = 0;
  virtual void read(uint64_t addr, void *dst, size_t size) = 0;
  virtual void write(uint64_t addr, const void *src, size_t size) = 0;
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
    freeBlocks = {std::make_pair(0, initialSize)};
    next = freeBlocks.begin();
  }

  /// Allocate 'size' bytes. We return an offset (uint64_t) into our single buffer.
  /// Throws std::bad_alloc if we can’t fit the allocation.
  uint64_t allocate(size_t size) override {
    // Keep track of first examined block
    auto oldNext = next;
    do {
        if (next->second >= size) {
            // Current block is large enough for allocation
            uint64_t addr = next->first;
            Allocation allocInfo;
            allocInfo.size = size;
            allocations[addr] = allocInfo;

            // Update block to reflect allocation
            next->first += size;
            next->second -= size;
            return addr;
        }
        // Advance list, circling back if at end
        next = ++next == freeBlocks.end() ? freeBlocks.begin() : next;
    } while (next != oldNext);

    // We've traversed all available blocks, and none are large enough
    throw std::bad_alloc();
  }

  /// Remove allocation associated with addr from map
  /// Also reclaims freed memory for future allocations
  void free(uint64_t addr) override {
    uint64_t freedSize = allocations[addr].size;
    allocations.erase(addr);

    // Consolidate blocks by reclaiming freed memory
    // Find supremum/infimum of freed block located at addr
    // (i.e. closest available blocks on either side of freed block)
    auto right = std::upper_bound(freeBlocks.begin(), freeBlocks.end(), addr,
        [](const uint64_t a, const std::pair<uint64_t, uint64_t>& b) {
            return a < b.first;
        }
    );
    auto left = std::prev(right);

    // Test if the adjacent blocks of the freed block are available
    // This determines how the blocks can be consolidated
    bool openLeft = left->first+left->second == addr;
    bool openRight = right->first == addr + freedSize;

    if (openLeft && openRight) {
        // Case I: Incorporate both the freed block and right into left
        left->second += (freedSize + right->second);
        // The right pointer is now redundant, so we can delete it
        // Ensure that next != right to prevent dangling pointer
        if (next == right) {
            next = ++next == freeBlocks.end() ? freeBlocks.begin() : next;
        }
        freeBlocks.erase(right); // made redundant by expansion
    }
    else if (openLeft && !openRight) {
        // Case II: Incorporate the freed block into left
        left->second += freedSize;
    }
    else if (!openLeft && openRight)  {
        // Case III: Incorporate the freed block into right
        right->first = addr;
        right->second += freedSize;
    }
    else {
        // Case IV: Bookended by allocated memory, create new entry between left and right
        auto newBlock = std::make_pair(addr, freedSize);
        freeBlocks.insert(right, newBlock);
    }

  }

  /// Read 'size' bytes from address 'addr' into 'dst'.
  /// Performs simple bounds checking against our recorded allocations.
  void read(uint64_t addr, void *dst, size_t size) override {
    auto [allocStart, allocInfo] = findAllocation(addr);
    // Ensure the entire read fits within [allocStart, allocStart + allocSize)
    if (addr + size > allocStart + allocInfo.size)
      throw std::runtime_error("SimpleMemoryManager: read out of bounds");

    std::memcpy(dst, &Mem[addr], size);
  }

  /// Write 'size' bytes from 'src' into address 'addr'.
  /// Performs simple bounds checking.
  void write(uint64_t addr, const void *src, size_t size) override {
    auto [allocStart, allocInfo] = findAllocation(addr);
    // Ensure the entire write fits within [allocStart, allocStart + allocSize)
    if (addr + size > allocStart + allocInfo.size)
      throw std::runtime_error("SimpleMemoryManager: write out of bounds");

    std::memcpy(&Mem[addr], src, size);
  }

private:
  /// A struct to track basic allocation metadata.
  /// TODO: Add alignment, original request, etc.
  struct Allocation {
    size_t size;
  };

  /// Lookup which allocation covers a given address 'addr'.
  /// We do a lower_bound or predecessor search to find the earliest allocation
  /// that starts before (or at) 'addr'.
  std::pair<uint64_t, Allocation> findAllocation(uint64_t addr) {
    // Upper bound returns the first iterator greater than 'addr'.
    auto it = allocations.upper_bound(addr);
    // If 'it' is not the first in the map, step back one to see if that
    // allocation covers 'addr'.
    if (it != allocations.begin()) {
      --it;
      uint64_t allocStart = it->first;
      const Allocation &info = it->second;
      if (addr >= allocStart && addr < (allocStart + info.size)) {
        return {allocStart, info};
      }
    }
    throw std::runtime_error("SimpleMemoryManager: address not in any allocation");
  }

  // A single contiguous buffer
  std::vector<char> Mem;

  // All available memory blocks, available as pairs [beginAddr, size]
  std::list<std::pair<uint64_t, uint64_t>> freeBlocks;

  // Iterator to the last allocated block in memory
  // Used for "next-block" allocation strategy
  std::list<std::pair<uint64_t, uint64_t>>::iterator next;

  // Map from "address" (offset) -> Allocation metadata
  std::map<uint64_t, Allocation> allocations;

};
}

// TODO: Eventaully add something for, say, a DRAMSimMemoryManager or other popular memory systems.

#endif // MLIR_INTERPRETER_MEMORYMANAGER_H
