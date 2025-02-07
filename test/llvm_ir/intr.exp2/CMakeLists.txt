if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_exp2.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_exp2.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_EXP2
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_exp2.mlir --args=3)
    add_test(
        NAME INTR_EXP2_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_exp2.bc --args=3)
    set_tests_properties(INTR_EXP2 INTR_EXP2_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "8")
endif()