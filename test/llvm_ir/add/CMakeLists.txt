if(ENABLE_TESTING)
  execute_process(
    COMMAND "/data/home/builduser/.local/opt/llvm-polygeist/bin/mlir-opt --emit-bytecode -o test_add.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_add.mlir"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RESULT_VARIABLE _mlir_rc
    OUTPUT_VARIABLE _mlir_stdout
    ERROR_VARIABLE _mlir_stderr
    COMMAND_ECHO STDOUT
    # (optional) strip trailing newlines:
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  # Print return code, stdout, and stderr to the CMake log
  message(STATUS "mlir-opt return code: ${_mlir_rc}")
  if(_mlir_stdout)
    message(STATUS "mlir-opt stdout:\n${_mlir_stdout}")
  endif()
  if(_mlir_stderr)
    message(WARNING "mlir-opt stderr:\n${_mlir_stderr}")
  endif()

#   add_test(
#     NAME ADD
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#     COMMAND mli test_add.mlir --args=1,2)
#   add_test(
#     NAME ADD_BYTECODE
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMAND mli test_add.bc --args=1,2)
#   set_tests_properties(
#     ADD ADD_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all" PASS_REGULAR_EXPRESSION
#                                 ".+result: .+3")
endif()
