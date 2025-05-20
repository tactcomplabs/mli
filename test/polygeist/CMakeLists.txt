if(ENABLE_TESTING)
  set(CGEIST ${POLYGEIST_SRC}/build/bin/cgeist)
  add_test(
    NAME POLYGEIST_C
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${CGEIST} matmul.c -S)
  add_test(
    NAME POLYGEIST_CXX
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${CGEIST} matmul.cpp -S)
  set_tests_properties(
    POLYGEIST_C POLYGEIST_CXX PROPERTIES TIMEOUT 60 LABELS "all"
                                         PASS_REGULAR_EXPRESSION "@[_Z0-9]*matmul")
endif()
