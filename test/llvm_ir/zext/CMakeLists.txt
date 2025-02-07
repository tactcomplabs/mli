if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_zext.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_zext.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ZEXT
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_zext.mlir --args=-1)
    add_test(
        NAME ZEXT_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_zext.bc --args=-1)
    set_tests_properties(ZEXT ZEXT_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "4294967295")
endif()