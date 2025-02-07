if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_pow.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_pow.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_POW
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_pow.mlir --args=2.25,0.5)
    add_test(
        NAME INTR_POW_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_pow.bc --args=2.25,0.5)
    set_tests_properties(INTR_POW INTR_POW_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "1.5")
endif()