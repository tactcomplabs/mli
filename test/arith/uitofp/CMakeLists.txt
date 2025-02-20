if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_uitofp.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_uitofp.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_UITOFP
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_uitofp.mlir --args=42)
    add_test(
        NAME ARITH_UITOFP_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_uitofp.bc --args=42)
    set_tests_properties(ARITH_UITOFP ARITH_UITOFP_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+42")
endif()
