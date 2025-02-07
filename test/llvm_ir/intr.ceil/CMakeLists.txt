if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_ceil.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_ceil.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_CEIL
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_ceil.mlir --args=1.5)
    add_test(
        NAME INTR_CEIL_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_ceil.bc --args=1.5)
    set_tests_properties(INTR_CEIL INTR_CEIL_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+2")
endif()