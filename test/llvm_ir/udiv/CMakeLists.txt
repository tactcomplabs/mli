if(ENABLE_TESTING)
   add_test(NAME UDIV
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_udiv.mlir --args=1,2
   )
   set_tests_properties(UDIV
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "0"
   )
endif()