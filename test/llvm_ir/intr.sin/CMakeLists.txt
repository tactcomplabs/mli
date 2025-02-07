if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_sin.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_sin.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_SIN
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_sin.mlir --args=0)
    add_test(
        NAME INTR_SIN_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_sin.bc --args=0)
    set_tests_properties(INTR_SIN INTR_SIN_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "0")
endif()