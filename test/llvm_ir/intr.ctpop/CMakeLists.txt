if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_ctpop.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_ctpop.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_CTPOP
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_ctpop.mlir --args=42)
    add_test(
        NAME INTR_CTPOP_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_ctpop.bc --args=42)
    set_tests_properties(INTR_CTPOP INTR_CTPOP_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "3")
endif()