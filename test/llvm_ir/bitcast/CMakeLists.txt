if(ENABLE_TESTING)
  execute_process(
    COMMAND mlir-opt --emit-bytecode -o test_bitcast.bc
            ${CMAKE_CURRENT_SOURCE_DIR}/test_bitcast.mlir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test(
    NAME BITCAST_INTTOFP
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_bitcast.mlir --args=1069547520 --func=itofp)
  add_test(
    NAME BITCAST_INTTOFP_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_bitcast.bc --args=1069547520 --func=itofp)
  add_test(
    NAME BITCAST_FPTOINT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND mli test_bitcast.mlir --args=1.5 --func=fptoi)
  add_test(
    NAME BITCAST_FPTOINT_BYTECODE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND mli test_bitcast.bc --args=1.5 --func=fptoi)
  set_tests_properties(
    BITCAST_INTTOFP BITCAST_INTTOFP_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                        PASS_REGULAR_EXPRESSION ".+result: .+1.5")
    set_tests_properties(
    BITCAST_FPTOINT BITCAST_FPTOINT_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                        PASS_REGULAR_EXPRESSION ".+result: .+1069547520")
endif()
