if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_bswap.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_bswap.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_BSWAP
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_bswap.mlir --args=100000)
    add_test(
        NAME INTR_BSWAP_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_bswap.bc --args=100000)
    set_tests_properties(INTR_BSWAP INTR_BSWAP_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "2693136640")
endif()