if(ENABLE_TESTING)
   add_test(NAME INTR.SQRT
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_sqrt.mlir --args=2025
   )
   set_tests_properties(INTR.SQRT
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "45"
   )
endif()