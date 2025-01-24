if(ENABLE_TESTING)
   add_test(NAME INTR.FABS
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_fabs.mlir --args=1.5
   )
   set_tests_properties(INTR.FABS
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1.5"
   )
endif()