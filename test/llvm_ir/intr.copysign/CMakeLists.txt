if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_copysign.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_copysign.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_COPYSIGN
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_copysign.mlir --args=1.5,-2.25)
    add_test(
        NAME INTR_COPYSIGN_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_copysign.bc --args=1.5,-2.25)
    set_tests_properties(INTR_COPYSIGN INTR_COPYSIGN_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "-1.5")
endif()