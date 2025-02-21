if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_remf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_remf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_REMF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_remf.mlir --args=7.5,2.0)
    add_test(
        NAME ARITH_REMF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_remf.bc --args=7.5,2.0)
    set_tests_properties(ARITH_REMF ARITH_REMF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1.5")
endif()
