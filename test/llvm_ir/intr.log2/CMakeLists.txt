if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_log2.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_log2.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_LOG2
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_log2.mlir --args=32)
    add_test(
        NAME INTR_LOG2_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_log2.bc --args=32)
    set_tests_properties(INTR_LOG2 INTR_LOG2_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "5")
endif()