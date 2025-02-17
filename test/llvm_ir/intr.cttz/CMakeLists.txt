if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_cttz.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_cttz.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_CTTZ
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_cttz.mlir --args=8)
    add_test(
        NAME INTR_CTTZ_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_cttz.bc --args=8)
    set_tests_properties(INTR_CTTZ INTR_CTTZ_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+3")
endif()