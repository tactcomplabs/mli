if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_shl.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_shl.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME SHL
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_shl.mlir --args=1,2)
    add_test(
        NAME SHL_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_shl.bc --args=1,2)
    set_tests_properties(SHL SHL_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "4")
endif()