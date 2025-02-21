if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_sitofp.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_sitofp.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ARITH_SITOFP
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_sitofp.mlir --args=-42)
    add_test(
        NAME ARITH_SITOFP_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_sitofp.bc --args=-42)
    set_tests_properties(ARITH_SITOFP ARITH_SITOFP_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+-42")
endif()
