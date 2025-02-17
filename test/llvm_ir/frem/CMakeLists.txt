if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_frem.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_frem.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME FREM
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_frem.mlir --args=1.5,2.25)
    add_test(
        NAME FREM_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_frem.bc --args=1.5,2.25)
    set_tests_properties(FREM FREM_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1.5")
endif()