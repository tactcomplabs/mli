if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_sdiv.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_sdiv.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME SDIV
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_sdiv.mlir --args=1,2)
    add_test(
        NAME SDIV_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_sdiv.bc --args=1,2)
    set_tests_properties(SDIV SDIV_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "0")
endif()