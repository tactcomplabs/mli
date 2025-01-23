if(ENABLE_TESTING)
   add_test(NAME INTR.MAXNUM
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_maxnum.mlir --args=1.5,2.25
   )
   set_tests_properties(INTR.MAXNUM
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 2.25"
   )
    add_test(NAME INTR.MAXNUM_NAN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_maxnum.mlir --args=1.5,NaN
   )
   set_tests_properties(INTR.MAXNUM_NAN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1.5"
   )
endif()