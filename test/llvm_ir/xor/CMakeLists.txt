if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_xor.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_xor.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME XOR
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_xor.mlir --args=1,2)
    add_test(
        NAME XOR_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_xor.bc --args=1,2)
    set_tests_properties(XOR XOR_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "3")
endif()