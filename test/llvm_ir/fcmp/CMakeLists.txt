if(ENABLE_TESTING)
  execute_process(
    COMMAND mlir-opt --emit-bytecode -o test_fcmp.bc
            ${CMAKE_CURRENT_SOURCE_DIR}/test_fcmp.mlir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test(
    NAME FCMP_OEQ
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=1.5,1.5 --func=oeq)
  add_test(
    NAME FCMP_ONE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=1.5,2 --func=one)
  add_test(
    NAME FCMP_OLT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=-1.0,1.1 --func=olt)
  add_test(
    NAME FCMP_OLE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=1,1.0 --func=ole)
  add_test(
    NAME FCMP_OGT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=1.0,-1.0 --func=ogt)
  add_test(
    NAME FCMP_OGE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=-2.2,-2.2 --func=oge)
  add_test(
    NAME FCMP_ORD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=NaN,1 --func=ord)
  add_test(
    NAME FCMP_UEQ
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=NaN,1 --func=ueq)
  add_test(
    NAME FCMP_UNE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=1,NaN --func=une)
  add_test(
    NAME FCMP_ULT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir "--args=NaN, NaN" --func=ult)
  add_test(
    NAME FCMP_ULE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=1,2 --func=ule)
  add_test(
    NAME FCMP_UGT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=1.1,1.0 --func=ugt)
  add_test(
    NAME FCMP_UGE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir --args=1.5,1.4 --func=uge)
  add_test(
    NAME FCMP_UNO
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_fcmp.mlir "--args=NaN, NaN" --func=uno)

  add_test(
    NAME FCMP_OEQ_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=1.5,1.5 --func=oeq)
  add_test(
    NAME FCMP_ONE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=1.5,2 --func=one)
  add_test(
    NAME FCMP_OLT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=-1.0,1.1 --func=olt)
  add_test(
    NAME FCMP_OLE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=1,1.0 --func=ole)
  add_test(
    NAME FCMP_OGT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=1.0,-1.0 --func=ogt)
  add_test(
    NAME FCMP_OGE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=-2.2,-2.2 --func=oge)
  add_test(
    NAME FCMP_ORD_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=NaN,1 --func=ord)
  add_test(
    NAME FCMP_UEQ_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=NaN,1 --func=ueq)
  add_test(
    NAME FCMP_UNE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=1,NaN --func=une)
  add_test(
    NAME FCMP_ULT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc "--args=NaN, NaN" --func=ult)
  add_test(
    NAME FCMP_ULE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=1,2 --func=ule)
  add_test(
    NAME FCMP_UGT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=1.1,1.0 --func=ugt)
  add_test(
    NAME FCMP_UGE_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc --args=1.5,1.4 --func=uge)
  add_test(
    NAME FCMP_UNO_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_fcmp.bc "--args=NaN, NaN" --func=uno)

  set(FCMP_TESTS
      FCMP_OEQ
      FCMP_ONE
      FCMP_OLT
      FCMP_OLE
      FCMP_OGT
      FCMP_OGE
      FCMP_OLT
      FCMP_OLE
      FCMP_OGT
      FCMP_OGE
      FCMP_UEQ
      FCMP_UNE
      FCMP_ULT
      FCMP_ULE
      FCMP_UGT
      FCMP_UGE
      FCMP_ULT
      FCMP_ULE
      FCMP_UGT
      FCMP_UGE
      FCMP_OEQ_BYTECODE
      FCMP_ONE_BYTECODE
      FCMP_OLT_BYTECODE
      FCMP_OLE_BYTECODE
      FCMP_OGT_BYTECODE
      FCMP_OGE_BYTECODE
      FCMP_OLT_BYTECODE
      FCMP_OLE_BYTECODE
      FCMP_OGT_BYTECODE
      FCMP_OGE_BYTECODE
      FCMP_UEQ_BYTECODE
      FCMP_UNE_BYTECODE
      FCMP_ULT_BYTECODE
      FCMP_ULE_BYTECODE
      FCMP_UGT_BYTECODE
      FCMP_UGE_BYTECODE
      FCMP_ULT_BYTECODE
      FCMP_ULE_BYTECODE
      FCMP_UGT_BYTECODE
      FCMP_UGE_BYTECODE)

foreach(test ${FCMP_TESTS})
  set_tests_properties(${test} PROPERTIES TIMEOUT 60 LABELS "all"
                                                PASS_REGULAR_EXPRESSION ".+result: .+1")
endforeach()
    

endif()
