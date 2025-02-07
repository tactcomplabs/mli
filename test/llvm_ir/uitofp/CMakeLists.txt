if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_uitofp.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_uitofp.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME UITOFP
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_uitofp.mlir --args=1)
    add_test(
        NAME UITOFP_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_uitofp.bc --args=1)
    set_tests_properties(UITOFP UITOFP_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "1")
endif()