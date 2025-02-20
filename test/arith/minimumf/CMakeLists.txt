if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_minimumf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_minimumf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_MINIMUMF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_minimumf.mlir --args=3.5,7.2)
    add_test(
        NAME ARITH_MINIMUMF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_minimumf.bc --args=3.5,7.2)
    set_tests_properties(ARITH_MINIMUMF ARITH_MINIMUMF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+3.5")
endif()
