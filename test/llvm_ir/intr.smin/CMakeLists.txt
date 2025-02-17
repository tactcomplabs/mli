if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_smin.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_smin.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_SMIN
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_smin.mlir --args=1,2)
    add_test(
        NAME INTR_SMIN_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_smin.bc --args=1,2)
    set_tests_properties(INTR_SMIN INTR_SMIN_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1")
endif()