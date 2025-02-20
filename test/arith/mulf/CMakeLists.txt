if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_mulf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_mulf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_MULF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_mulf.mlir --args=3.0,4.0)
    add_test(
        NAME ARITH_MULF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_mulf.bc --args=3.0,4.0)
    set_tests_properties(ARITH_MULF ARITH_MULF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+12")
endif()
