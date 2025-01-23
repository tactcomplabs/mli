if(ENABLE_TESTING)
   add_test(NAME INTR.CEIL
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_ceil.mlir --args=1.5
   )
   set_tests_properties(INTR.CEIL
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 2"
   )
endif()