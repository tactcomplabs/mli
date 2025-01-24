if(ENABLE_TESTING)
   add_test(NAME SUB
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_sub.mlir --args=1,2
   )
   set_tests_properties(SUB
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "-1"
   )
endif()