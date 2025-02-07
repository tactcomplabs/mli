if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_fptosi.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_fptosi.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME FPTOSI
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_fptosi.mlir --args=1.5)
    add_test(
        NAME FPTOSI_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_fptosi.bc --args=1.5)
    set_tests_properties(FPTOSI FPTOSI_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "1")
endif()