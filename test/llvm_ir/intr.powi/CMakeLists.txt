if(ENABLE_TESTING)
   add_test(NAME INTR.POWI
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_powi.mlir --args=1.5,2
   )
   set_tests_properties(INTR.POWI
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "2.25"
   )
endif()