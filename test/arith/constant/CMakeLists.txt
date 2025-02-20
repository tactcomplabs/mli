if(ENABLE_TESTING)
  execute_process(
    COMMAND mlir-opt --emit-bytecode -o test_arith_constant.bc
            ${CMAKE_CURRENT_SOURCE_DIR}/test_arith_constant.mlir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_test(
    NAME ARITH_CONSTANT_INT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_arith_constant.mlir --func=int)

  add_test(
    NAME ARITH_CONSTANT_INT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_arith_constant.bc --func=int)

  set_tests_properties(
    ARITH_CONSTANT_INT ARITH_CONSTANT_INT_BYTECODE
    PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION
               "Initializing constant 42")
  add_test(
    NAME ARITH_CONSTANT_FLOAT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_arith_constant.mlir --func=float)
  add_test(
    NAME ARITH_CONSTANT_FLOAT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_arith_constant.bc --func=float)
  set_tests_properties(
    ARITH_CONSTANT_FLOAT ARITH_CONSTANT_FLOAT_BYTECODE
    PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION
               "Initializing constant 2.5")
endif()
