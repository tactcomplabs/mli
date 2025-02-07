if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_fma.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_fma.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_FMA
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_fma.mlir --args=1.5,2.25,3.125)
    add_test(
        NAME INTR_FMA_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_fma.bc --args=1.5,2.25,3.125)
    set_tests_properties(INTR_FMA INTR_FMA_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "6.5")
endif()