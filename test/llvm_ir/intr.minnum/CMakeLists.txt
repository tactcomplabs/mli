if(ENABLE_TESTING)
  execute_process(
    COMMAND mlir-opt --emit-bytecode -o test_intr_minnum.bc
            ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_minnum.mlir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test(
    NAME INTR_MINNUM
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_intr_minnum.mlir --args=1.5,2.25)
  add_test(
    NAME INTR_MINNUM_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_intr_minnum.bc --args=1.5,2.25)
  set_tests_properties(
    INTR_MINNUM INTR_MINNUM_BYTECODE
    PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION ".+result: .+ 1.5")
  add_test(
    NAME INTR_MINNUM_NAN
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_intr_minnum.mlir --args=1,NaN)
  add_test(
    NAME INTR_MINNUM_NAN_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_intr_minnum.bc --args=1,NaN)
  set_tests_properties(
    INTR_MINNUM_NAN INTR_MINNUM_NAN_BYTECODE
    PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION ".+result: .+ 1")
endif()
