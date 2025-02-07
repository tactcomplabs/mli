if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_umax.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_umax.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_UMAX
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_umax.mlir --args=1,2)
    add_test(
        NAME INTR_UMAX_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_umax.bc --args=1,2)
    set_tests_properties(INTR_UMAX INTR_UMAX_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "2")
endif()