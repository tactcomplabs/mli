if(ENABLE_TESTING)
   add_test(NAME INTR.ABS
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_abs.mlir --args=1
   )
   set_tests_properties(INTR.ABS
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1"
   )
endif()