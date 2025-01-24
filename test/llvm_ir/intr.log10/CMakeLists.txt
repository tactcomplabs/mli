if(ENABLE_TESTING)
   add_test(NAME INTR.LOG10
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_log10.mlir --args=100
   )
   set_tests_properties(INTR.LOG10
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "2"
   )
endif()