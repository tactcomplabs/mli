if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_bitreverse.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_bitreverse.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_BITREVERSE
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_bitreverse.mlir --args=3)
    add_test(
        NAME INTR_BITREVERSE_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_bitreverse.bc --args=3)
    set_tests_properties(INTR_BITREVERSE INTR_BITREVERSE_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+3221225472")
endif()