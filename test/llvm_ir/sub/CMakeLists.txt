if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_sub.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_sub.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME SUB
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_sub.mlir --args=1,2)
    add_test(
        NAME SUB_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_sub.bc --args=1,2)
    set_tests_properties(SUB SUB_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "-1")
endif()