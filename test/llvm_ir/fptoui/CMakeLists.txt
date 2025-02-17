if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_fptoui.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_fptoui.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME FPTOUI
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_fptoui.mlir --args=1.5)
    add_test(
        NAME FPTOUI_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_fptoui.bc --args=1.5)
    set_tests_properties(FPTOUI FPTOUI_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1")
endif()