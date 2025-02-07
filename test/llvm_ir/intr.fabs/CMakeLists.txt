if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_fabs.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_fabs.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_FABS
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_fabs.mlir --args=1.5)
    add_test(
        NAME INTR_FABS_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_fabs.bc --args=1.5)
    set_tests_properties(INTR_FABS INTR_FABS_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "1.5")
endif()