if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_fadd.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_fadd.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME FADD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_fadd.mlir --args=1.5,2.25)
    add_test(
        NAME FADD_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_fadd.bc --args=1.5,2.25)
    set_tests_properties(FADD FADD_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "3.75")
endif()