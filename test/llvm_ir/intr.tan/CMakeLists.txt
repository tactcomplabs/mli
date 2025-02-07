if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_tan.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_tan.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_TAN
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_tan.mlir --args=1.5)
    add_test(
        NAME INTR_TAN_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_tan.bc --args=1.5)
    set_tests_properties(INTR_TAN INTR_TAN_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "14.10141994717172")
endif()