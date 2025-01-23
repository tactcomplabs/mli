if(ENABLE_TESTING)
   add_test(NAME OR
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_or.mlir --args=1,2
   )
   set_tests_properties(OR
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 3"
   )
endif()