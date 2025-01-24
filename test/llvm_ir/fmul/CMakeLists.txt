if(ENABLE_TESTING)
   add_test(NAME FMUL
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_fmul.mlir --args=1.5,2.25
   )
   set_tests_properties(FMUL
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "3.375"
   )
endif()