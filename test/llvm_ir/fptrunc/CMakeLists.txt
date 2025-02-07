if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_fptrunc.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_fptrunc.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME FPTRUNC
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_fptrunc.mlir --args=3.14159268)
    add_test(
        NAME FPTRUNC_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_fptrunc.bc --args=3.14159268)
    set_tests_properties(FPTRUNC FPTRUNC_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+3.1415925")
endif()