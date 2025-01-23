if(ENABLE_TESTING)
   add_test(NAME MUL
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_mul.mlir --args=1,2
   )
   set_tests_properties(MUL
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 2"
   )
endif()