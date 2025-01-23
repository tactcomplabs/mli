if(ENABLE_TESTING)
   add_test(NAME UREM
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_urem.mlir --args=1,2
   )
   set_tests_properties(UREM
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1"
   )
endif()