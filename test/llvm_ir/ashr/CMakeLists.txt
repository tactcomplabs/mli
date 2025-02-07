if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_ashr.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_ashr.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ASHR
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_ashr.mlir --args=1,2)
    add_test(
        NAME ASHR_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_ashr.bc --args=1,2)
    set_tests_properties(ASHR ASHR_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+0")
endif()