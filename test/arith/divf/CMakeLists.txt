if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_divf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_divf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_DIVF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_divf.mlir --args=10.0,2.0)
    add_test(
        NAME ARITH_DIVF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_divf.bc --args=10.0,2.0)
    set_tests_properties(ARITH_DIVF ARITH_DIVF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+5")
endif()
