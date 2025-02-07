if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_fdiv.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_fdiv.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME FDIV
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_fdiv.mlir --args=1.0,2.0)
    add_test(
        NAME FDIV_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_fdiv.bc --args=1.0,2.0)
    set_tests_properties(FDIV FDIV_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "0.5")
endif()