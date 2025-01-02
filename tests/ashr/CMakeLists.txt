if(ENABLE_TESTING)
   add_test(NAME ASHR
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_ashr.mlir --args=1,2
   )
   set_tests_properties(ASHR
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 0"
   )
endif()