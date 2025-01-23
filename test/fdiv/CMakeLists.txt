if(ENABLE_TESTING)
   add_test(NAME FDIV
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_fdiv.mlir --args=1.0,2.0
   )
   set_tests_properties(FDIV
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 0.5"
   )
endif()
