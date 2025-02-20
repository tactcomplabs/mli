if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_fptosi.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_fptosi.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_FPTOSI
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_fptosi.mlir --args=42.8)
    add_test(
        NAME ARITH_FPTOSI_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_fptosi.bc --args=42.8)
    set_tests_properties(ARITH_FPTOSI ARITH_FPTOSI_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+42")
endif()
