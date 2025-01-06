#ifndef MLIR_INTERPRETER_MEMORYMANAGER_H
#define MLIR_INTERPRETER_MEMORYMANAGER_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <new>
#include <stdexcept>

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
  explicit SimpleMemoryManager(size_t initialSize = 1024 * 1024)
      : nextFreeAddress(0) {
    // Pre-allocate our contiguous memory space
    Mem.resize(initialSize, 0);
  }

  /// Allocate 'size' bytes. We return an offset (uint64_t) into our single buffer.
  /// Throws std::bad_alloc if we can’t fit the allocation.
  uint64_t allocate(size_t size) override {
    // Check if we have room in 'mem'
    if (nextFreeAddress + size > Mem.size())
      throw std::bad_alloc();

    uint64_t addr = nextFreeAddress;
    nextFreeAddress += size;

    Allocation allocInfo;
    allocInfo.size = size;
    allocations[addr] = allocInfo;

    return addr;
  }

  /// Free is a no-op in this naive implementation, but we do erase the allocation
  /// from our map to prevent bounds checking from succeeding if a region is freed.
  void free(uint64_t addr) override {
    auto it = allocations.find(addr);
    if (it != allocations.end()) {
      allocations.erase(it);
    }
    // TODO: Eventually add support for reusing freed memory.
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

  // The offset for the next allocation.
  uint64_t nextFreeAddress;

  // Map from "address" (offset) -> Allocation metadata
  std::map<uint64_t, Allocation> allocations;
};
}

// TODO: Eventaully add something for, say, a DRAMSimMemoryManager or other popular memory systems.

#endif // MLIR_INTERPRETER_MEMORYMANAGER_H
