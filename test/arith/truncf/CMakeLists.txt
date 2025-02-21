if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_truncf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_truncf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_TRUNCF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_truncf.mlir --args=3.14159265359)
    add_test(
        NAME ARITH_TRUNCF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_truncf.bc --args=3.14159265359)
    set_tests_properties(ARITH_TRUNCF ARITH_TRUNCF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+3.14159")
endif()
