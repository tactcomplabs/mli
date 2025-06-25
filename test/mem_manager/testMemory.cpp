#include "mlir/Interpreter/MemoryManager.h"
#include <cassert>
#include <iostream>

#define NUM_BLOCKS 3

using namespace mlir;

// We need to test if consolidating adjacent free blocks works
// Do this by testing if we can allocate a region spanning the width of combined blocks
// Throws an uncaught exception if allocate's exception and throwException disagree
uint64_t testAlloc(SimpleMemoryManager& mem, const uint64_t size, bool throwException = false) {
    uint64_t addr;
    try {
        addr = mem.allocate(size);
        // alloc doesn't throw exception, determine if this is good or bad
        if ( throwException )
            throw std::runtime_error("No exception thrown!");
        mem.free(addr);
    } catch ( std::exception& e ) {
        // Throw a second exception if the test fails
        if ( !throwException || std::string(e.what()) == "No exception thrown" )
            throw std::exception();
    }
    return addr;
}

// Set up memory manager to model each of the four removal scenarios
void prepareMemManager(SimpleMemoryManager& mem, const uint64_t* blockSizes, bool openLeft, bool openRight) {
    uint64_t addresses[NUM_BLOCKS];

    for ( int i = 0; i < NUM_BLOCKS; i++ ) {
        addresses[i] = mem.allocate(blockSizes[i]);
    }

    if ( openLeft )
        mem.free(addresses[0]);
    if ( openRight )
        mem.free(addresses[2]);
}

int main() {
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

    const uint64_t middleAddr = 20;

    // Case I
    mem1.free(middleAddr);
    assert(testAlloc(mem1, blockSizes[1], false) == middleAddr && "MemManager can't reclaim freed block");
    testAlloc(mem1, memSize, true);  // should fail, not enough space

    // Case II
    mem2.free(middleAddr);
    assert(testAlloc(mem2, blockSizes[1], false) == middleAddr && "MemManager can't reclaim freed block");
    assert(
        testAlloc(mem2, blockSizes[1] + blockSizes[2], false) == middleAddr &&
        "MemManager can't consolidate block with right neighbor"
    );
    testAlloc(mem2, memSize, true);  // should fail, not enough space

    // Case III
    mem3.free(middleAddr);
    assert(testAlloc(mem3, blockSizes[1], false) == middleAddr - blockSizes[0] && "MemManager can't reclaim freed block");
    assert(
        testAlloc(mem3, blockSizes[0] + blockSizes[1], false) == middleAddr - blockSizes[0] &&
        "MemManager can't consolidate block with left neighbor"
    );
    testAlloc(mem3, memSize, true);  // should fail, not enough space

    // Case IV
    mem4.free(middleAddr);
    assert(testAlloc(mem4, blockSizes[1], false) == 0 && "MemManager can't reclaim freed block");
    assert(testAlloc(mem4, memSize, false) == 0 && "MemManager can't consolidate block with both neighbors");

    std::cout << "SUCCESS" << std::endl;
    return 0;
}
