if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_srem.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_srem.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME SREM
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_srem.mlir --args=1,2)
    add_test(
        NAME SREM_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_srem.bc --args=1,2)
    set_tests_properties(SREM SREM_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1")
endif()