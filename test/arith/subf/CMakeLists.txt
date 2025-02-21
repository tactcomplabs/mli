if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_subf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_subf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_SUBF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_subf.mlir --args=7.5,2.5)
    add_test(
        NAME ARITH_SUBF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_subf.bc --args=7.5,2.5)
    set_tests_properties(ARITH_SUBF ARITH_SUBF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+5")
endif()
