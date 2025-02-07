if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_udiv.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_udiv.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME UDIV
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_udiv.mlir --args=1,2)
    add_test(
        NAME UDIV_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_udiv.bc --args=1,2)
    set_tests_properties(UDIV UDIV_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "0")
endif()