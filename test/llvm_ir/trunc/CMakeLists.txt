if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_trunc.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_trunc.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME TRUNC
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_trunc.mlir --args=9223372036854775807)
    add_test(
        NAME TRUNC_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_trunc.bc --args=9223372036854775807)
    set_tests_properties(TRUNC TRUNC_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "-1")
endif()