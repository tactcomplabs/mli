if(ENABLE_TESTING)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_addi.bc
    COMMAND mlir-opt --emit-bytecode -o test_addi.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_addi.mlir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating test_addi.bc from test_addi.mlir"
    VERBATIM
  )

  add_custom_target(generate_test_addi_bc
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test_addi.bc
  )
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
