if(ENABLE_TESTING)
   add_test(NAME SREM
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_srem.mlir --args=1,2
   )
   set_tests_properties(SREM
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1"
   )
endif()