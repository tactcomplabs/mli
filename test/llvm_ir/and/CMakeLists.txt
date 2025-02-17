if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_and.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_and.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME AND
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_and.mlir --args=1,2)
    add_test(
        NAME AND_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_and.bc --args=1,2)
    set_tests_properties(AND AND_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+0")
endif()