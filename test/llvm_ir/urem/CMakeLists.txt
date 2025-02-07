if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_urem.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_urem.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME UREM
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_urem.mlir --args=1,2)
    add_test(
        NAME UREM_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_urem.bc --args=1,2)
    set_tests_properties(UREM UREM_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "1")
endif()