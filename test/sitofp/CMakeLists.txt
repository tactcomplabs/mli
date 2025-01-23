if(ENABLE_TESTING)
   add_test(NAME SITOFP
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_sitofp.mlir --args=1
   )
   set_tests_properties(SITOFP
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1"
   )
endif()