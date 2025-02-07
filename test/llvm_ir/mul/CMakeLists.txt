if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_mul.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_mul.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME MUL
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_mul.mlir --args=1,2)
    add_test(
        NAME MUL_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_mul.bc --args=1,2)
    set_tests_properties(MUL MUL_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "2")
endif()