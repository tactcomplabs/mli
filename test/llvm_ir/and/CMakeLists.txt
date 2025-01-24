if(ENABLE_TESTING)
   add_test(NAME AND
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_and.mlir --args=1,2
   )
   set_tests_properties(AND
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "0"
   )
endif()