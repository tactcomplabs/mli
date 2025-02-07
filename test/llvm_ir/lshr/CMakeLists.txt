if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_lshr.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_lshr.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME LSHR
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_lshr.mlir --args=1,2)
    add_test(
        NAME LSHR_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_lshr.bc --args=1,2)
    set_tests_properties(LSHR LSHR_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "0")
endif()