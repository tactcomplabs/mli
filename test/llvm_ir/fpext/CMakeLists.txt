if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_fpext.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_fpext.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME FPEXT
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_fpext.mlir --args=1.5)
    add_test(
        NAME FPEXT_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_fpext.bc --args=1.5)
    set_tests_properties(FPEXT FPEXT_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "1.5")
endif()