#include "mlir/Interpreter/MemoryManager.h"
#include <cassert>
#include <iostream>

#define NUM_BLOCKS 3

using namespace mlir;

// We need to test if consolidating adjacent free blocks works
// Do this by testing if we can allocate a region spanning the width of combined blocks
// Throws an uncaught exception if allocate's exception and throwException disagree
void testAlloc(SimpleMemoryManager& mem, const uint64_t size, bool throwException = false) {
    uint64_t id;
    try {
        id = mem.allocate(size);
        // alloc doesn't throw exception, determine if this is good or bad
        if ( throwException )
            throw std::runtime_error("No exception thrown!");
        mem.free(id);
    } catch ( std::exception& e ) {
        // Throw a second exception if the test fails
        if ( !throwException || std::string(e.what()) == "No exception thrown" )
            throw std::exception();
    }
}

void testRealloc(
    SimpleMemoryManager& mem, const uint64_t id, const uint64_t new_size, bool expectCopy = false, bool throwException = false
) {
    try {
        bool copy = mem.realloc(id, new_size);
        if ( copy != expectCopy ) {
            throw std::runtime_error("Copy semantics differ");
        }
        // realloc doesn't throw exception, determine if this is good or bad
        if ( throwException )
            throw std::runtime_error("No exception thrown!");
    } catch ( std::exception& e ) {
        // Throw a second exception if the test fails (i.e. we are not expecting an exception)
        if ( !throwException )
            throw e;
    }
}

// Set up memory manager to model each of the four removal scenarios
void prepareMemManager(SimpleMemoryManager& mem, const uint64_t* blockSizes, bool openLeft, bool openRight) {
    uint64_t alloc_ids[NUM_BLOCKS];

    for ( int i = 0; i < NUM_BLOCKS; i++ ) {
        alloc_ids[i] = mem.allocate(blockSizes[i]);
    }

    if ( openLeft )
        mem.free(alloc_ids[0]);
    if ( openRight )
        mem.free(alloc_ids[2]);
}

void runAllocTests() {
    const uint64_t memSize                = 100;
    const uint64_t blockSizes[NUM_BLOCKS] = {20, 50, 30};

    // Case I: Free a region blocked on both sides, no consolidation of freed blocks
    SimpleMemoryManager mem1              = SimpleMemoryManager(memSize);
    prepareMemManager(mem1, &blockSizes[0], false, false);

    // Case II: Free a region blocked only on left side, consolidate freed block w/ right neighbor
    SimpleMemoryManager mem2 = SimpleMemoryManager(memSize);
    prepareMemManager(mem2, &blockSizes[0], false, true);

    // Case III: Free a region blocked only on right side, consolidate freed block w/ left neighbor
    SimpleMemoryManager mem3 = SimpleMemoryManager(memSize);
    prepareMemManager(mem3, &blockSizes[0], true, false);

    // Case IV: Free a region blocked on neither side, consolidate freed block w/ both neighbors
    SimpleMemoryManager mem4 = SimpleMemoryManager(memSize);
    prepareMemManager(mem4, &blockSizes[0], true, true);

    const uint64_t middle_vaddr = 1ull << 32;

    // Case I
    mem1.free(middle_vaddr);
    testAlloc(mem1, blockSizes[1], false);  // should succeed, use original block
    testAlloc(mem1, memSize, true);         // should fail, not enough space

    // Case II
    mem2.free(middle_vaddr);
    testAlloc(mem2, blockSizes[1], false);                  // should succeed, reclaim block 1
    testAlloc(mem2, blockSizes[1] + blockSizes[2], false);  // should succeed, reclaim block 1+2
    testAlloc(mem2, memSize, true);                         // should fail, not enough space

    // Case III
    mem3.free(middle_vaddr);
    testAlloc(mem3, blockSizes[1], false);                  // should succeed, reclaim block 0
    testAlloc(mem3, blockSizes[0] + blockSizes[1], false);  // should succeed, reclaim block 0+1
    testAlloc(mem3, memSize, true);                         // should fail, not enough space

    // Case IV
    mem4.free(middle_vaddr);
    testAlloc(mem4, blockSizes[1], false);
    testAlloc(mem4, memSize, false);
}

void runReallocTests() {
    const uint64_t memSize                = 100;
    const uint64_t blockSizes[NUM_BLOCKS] = {20, 50, 30};

    // Case I: New size is less than old size, no copy
    SimpleMemoryManager mem1              = SimpleMemoryManager(memSize);
    uint64_t            id_1              = mem1.allocate(blockSizes[0]);
    testRealloc(mem1, id_1, 10, false, false);  //

    // Case II: New size is larger, merge with neighbor block, no copy
    SimpleMemoryManager mem2 = SimpleMemoryManager(memSize);
    uint64_t            id_2 = mem2.allocate(blockSizes[0]);
    testRealloc(mem2, id_2, 30, false, false);

    // Case III: New size is larger, can't merge with neighbor, must copy to new addr
    SimpleMemoryManager mem3 = SimpleMemoryManager(memSize);
    uint64_t            id_3 = mem3.allocate(blockSizes[0]);
    mem3.allocate(blockSizes[1]);
    testRealloc(mem3, id_3, blockSizes[2], true, false);
}

int main() {
    runAllocTests();
    runReallocTests();
    std::cout << "SUCCESS" << std::endl;
    return 0;
}
