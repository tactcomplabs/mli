if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_umin.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_umin.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_UMIN
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_umin.mlir --args=1,2)
    add_test(
        NAME INTR_UMIN_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_umin.bc --args=1,2)
    set_tests_properties(INTR_UMIN INTR_UMIN_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "1")
endif()