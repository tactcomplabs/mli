if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_cmpi.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_cmpi.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_CMPI
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_cmpi.mlir --args=20,10)
    add_test(
        NAME ARITH_CMPI_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_cmpi.bc --args=20,10)
    set_tests_properties(ARITH_CMPI ARITH_CMPI_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1")
endif()
