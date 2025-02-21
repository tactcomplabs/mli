if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_maximumf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_maximumf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_MAXIMUMF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_maximumf.mlir --args=3.5,7.2)
    add_test(
        NAME ARITH_MAXIMUMF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_maximumf.bc --args=3.5,7.2)
    set_tests_properties(ARITH_MAXIMUMF ARITH_MAXIMUMF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+7.2")
endif()
