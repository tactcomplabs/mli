if(ENABLE_TESTING)
   add_test(NAME FPTOSI
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_fptosi.mlir --args=1.5
   )
   set_tests_properties(FPTOSI
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1"
   )
endif()