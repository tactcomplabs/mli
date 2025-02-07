if(ENABLE_TESTING)
  execute_process(
    COMMAND mlir-opt --emit-bytecode -o test_icmp.bc
            ${CMAKE_CURRENT_SOURCE_DIR}/test_icmp.mlir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test(
    NAME ICMP_EQ
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=1,1 --func=eq)
  add_test(
    NAME ICMP_NE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=1,2 --func=ne)
  add_test(
    NAME ICMP_SLT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=-1,1 --func=slt)
  add_test(
    NAME ICMP_SLE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=1,1 --func=sle)
  add_test(
    NAME ICMP_SGT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=-1,-2 --func=sgt)
  add_test(
    NAME ICMP_SGE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=-2,-2 --func=sge)
  add_test(
    NAME ICMP_ULT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=1,2 --func=ult)
  add_test(
    NAME ICMP_ULE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=4294967295,-1 --func=ule)
  add_test(
    NAME ICMP_UGT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=-1,1 --func=ugt)
  add_test(
    NAME ICMP_UGE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_icmp.mlir --args=-1,-2 --func=uge)

  add_test(
    NAME ICMP_EQ_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=1,1 --func=eq)
  add_test(
    NAME ICMP_NE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=1,2 --func=ne)
  add_test(
    NAME ICMP_SLT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=-1,1 --func=slt)
  add_test(
    NAME ICMP_SLE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=1,1 --func=sle)
  add_test(
    NAME ICMP_SGT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=-1,-2 --func=sgt)
  add_test(
    NAME ICMP_SGE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=-2,-2 --func=sge)
  add_test(
    NAME ICMP_ULT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=1,2 --func=ult)
  add_test(
    NAME ICMP_ULE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=4294967295,-1 --func=ule)
  add_test(
    NAME ICMP_UGT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=-1,1 --func=ugt)
  add_test(
    NAME ICMP_UGE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_icmp.bc --args=-1,-2 --func=uge)

  set(ICMP_TESTS
      ICMP_EQ
      ICMP_NE
      ICMP_SLT
      ICMP_SLE
      ICMP_SGT
      ICMP_SGE
      ICMP_ULT
      ICMP_ULE
      ICMP_UGT
      ICMP_UGE
      ICMP_EQ_BYTECODE
      ICMP_NE_BYTECODE
      ICMP_SLT_BYTECODE
      ICMP_SLE_BYTECODE
      ICMP_SGT_BYTECODE
      ICMP_SGE_BYTECODE
      ICMP_ULT_BYTECODE
      ICMP_ULE_BYTECODE
      ICMP_UGT_BYTECODE
      ICMP_UGE_BYTECODE)

  foreach(test ${ICMP_TESTS})
    set_tests_properties(
      ${test} PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION
                         ".+result: .+1")
  endforeach()

endif()
