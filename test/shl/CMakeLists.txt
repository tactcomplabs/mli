if(ENABLE_TESTING)
   add_test(NAME SHL
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_shl.mlir --args=1,2
   )
   set_tests_properties(SHL
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 4"
   )
endif()