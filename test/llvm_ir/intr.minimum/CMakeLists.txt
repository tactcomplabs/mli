if(ENABLE_TESTING)
   add_test(NAME INTR.MINIMUM
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_minimum.mlir --args=1.5,2.25
   )
   set_tests_properties(INTR.MINIMUM
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1.5"
   )
    add_test(NAME INTR.MINIMUM_NAN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_minimum.mlir --args=1.5,NaN
   )
   set_tests_properties(INTR.MINIMUM_NAN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "NaN"
   )
endif()