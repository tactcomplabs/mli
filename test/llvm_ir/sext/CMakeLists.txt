if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_sext.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_sext.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME SEXT
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_sext.mlir --args=1)
    add_test(
        NAME SEXT_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_sext.bc --args=1)
    set_tests_properties(SEXT SEXT_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1")
endif()