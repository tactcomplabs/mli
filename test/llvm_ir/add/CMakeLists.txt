if(ENABLE_TESTING)
  execute_process(
    COMMAND mlir-opt --emit-bytecode -o test_add.bc "${CMAKE_CURRENT_SOURCE_DIR}/test_add.mlir"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  add_test(
    NAME ADD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_add.mlir --args=1,2)
  add_test(
    NAME ADD_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_add.bc --args=1,2)
  set_tests_properties(
    ADD ADD_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION
                                ".+result: .+3")
endif()
