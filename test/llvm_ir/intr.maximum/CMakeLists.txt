if(ENABLE_TESTING)
  execute_process(
    COMMAND mlir-opt --emit-bytecode -o test_intr_maximum.bc
            ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_maximum.mlir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test(
    NAME INTR_MAXIMUM
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_intr_maximum.mlir --args=1.5,2.25)
  add_test(
    NAME INTR_MAXIMUM_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_intr_maximum.bc --args=1.5,2.25)
  set_tests_properties(
    INTR_MAXIMUM INTR_MAXIMUM_BYTECODE
    PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION ".+result: .+2.25")
  add_test(
    NAME INTR_MAXIMUM_NAN
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_intr_maximum.mlir --args=1,NaN)
  add_test(
    NAME INTR_MAXIMUM_NAN_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_intr_maximum.bc --args=1,NaN)
  set_tests_properties(
    INTR_MAXIMUM_NAN INTR_MAXIMUM_NAN_BYTECODE
    PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION ".+result: .+NaN")
endif()
