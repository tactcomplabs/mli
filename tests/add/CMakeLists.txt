if(ENABLE_TESTING)
   add_test(NAME ADD
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_add.mlir --args=1,2
   )
   set_tests_properties(ADD
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 3"
   )
endif()