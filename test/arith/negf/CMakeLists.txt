if(ENABLE_TESTING)
  execute_process(COMMAND mlir-opt --emit-bytecode -o test_negf.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_negf.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test(
        NAME ARITH_NEGF
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_negf.mlir --args=3.5)
  add_test(
        NAME ARITH_NEGF_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_negf.bc --args=3.5)
  set_tests_properties(ARITH_NEGF ARITH_NEGF_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+-3.5")
endif()
