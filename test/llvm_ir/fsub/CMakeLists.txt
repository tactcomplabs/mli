if(ENABLE_TESTING)
   add_test(NAME FSUB
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_fsub.mlir --args=1.5,2.25
   )
   set_tests_properties(FSUB
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "-0.75"
   )
endif()