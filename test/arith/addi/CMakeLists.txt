if(ENABLE_TESTING)
  execute_process(COMMAND mlir-opt --emit-bytecode -o test_addi.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_addi.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test(
        NAME ARITH_ADDI
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_addi.mlir --args=10,15)
  add_test(
        NAME ARITH_ADDI_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_addi.bc --args=10,15)
  set_tests_properties(ARITH_ADDI ARITH_ADDI_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+25")
endif()
