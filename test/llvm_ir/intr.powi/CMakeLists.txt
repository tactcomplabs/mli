if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_powi.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_powi.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_POWI
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_powi.mlir --args=1.5,2)
    add_test(
        NAME INTR_POWI_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_powi.bc --args=1.5,2)
    set_tests_properties(INTR_POWI INTR_POWI_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+2.25")
endif()