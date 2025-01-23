if(ENABLE_TESTING)
   add_test(NAME INTR.MAXIMUM
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_maximum.mlir --args=1.5,2.25
   )
   set_tests_properties(INTR.MAXIMUM
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 2.25"
   )
    add_test(NAME INTR.MAXIMUM_NAN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_maximum.mlir --args=1,NaN
   )
   set_tests_properties(INTR.MAXIMUM_NAN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: NaN"
   )
endif()