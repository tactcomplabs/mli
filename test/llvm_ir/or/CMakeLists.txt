if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_or.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_or.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME OR
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_or.mlir --args=1,2)
    add_test(
        NAME OR_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_or.bc --args=1,2)
    set_tests_properties(OR OR_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+3")
endif()