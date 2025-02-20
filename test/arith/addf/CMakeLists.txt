if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_addf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_addf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_ADDF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_addf.mlir --args=1.5,2.25)
    add_test(
        NAME ARITH_ADDF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_addf.bc --args=1.5,2.25)
    set_tests_properties(ARITH_ADDF ARITH_ADDF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+3.75")
endif()
