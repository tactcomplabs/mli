if(ENABLE_TESTING)
   add_test(NAME INTR.MINNUM
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_minnum.mlir --args=1.5,2.25
   )
   set_tests_properties(INTR.MINNUM
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1.5"
   )
    add_test(NAME INTR.MINNUM_NAN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_minnum.mlir --args=1.5,NaN
   )
   set_tests_properties(INTR.MINNUM_NAN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1.5"
   )
endif()