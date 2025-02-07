if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_fneg.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_fneg.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME FNEG
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_fneg.mlir --args=1.5)
    add_test(
        NAME FNEG_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_fneg.bc --args=1.5)
    set_tests_properties(FNEG FNEG_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "-1.5")
endif()