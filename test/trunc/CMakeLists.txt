if(ENABLE_TESTING)
   add_test(NAME TRUNC
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_trunc.mlir --args=9223372036854775807
   )
   set_tests_properties(TRUNC
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: -1"
   )
endif()