if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_sqrt.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_sqrt.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_SQRT
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_sqrt.mlir --args=2025)
    add_test(
        NAME INTR_SQRT_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_sqrt.bc --args=2025)
    set_tests_properties(INTR_SQRT INTR_SQRT_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "45")
endif()