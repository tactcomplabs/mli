if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_cmpf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_cmpf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_CMPF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_cmpf.mlir --args=2.5,1.5)
    add_test(
        NAME ARITH_CMPF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_cmpf.bc --args=2.5,1.5)
    set_tests_properties(ARITH_CMPF ARITH_CMPF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1")
endif()
