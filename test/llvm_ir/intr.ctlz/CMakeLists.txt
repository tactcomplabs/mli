if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_ctlz.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_ctlz.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_CTLZ
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_ctlz.mlir --args=1)
    add_test(
        NAME INTR_CTLZ_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_ctlz.bc --args=1)
    set_tests_properties(INTR_CTLZ INTR_CTLZ_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "31")
endif()