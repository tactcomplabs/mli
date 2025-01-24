if(ENABLE_TESTING)
   add_test(NAME UITOFP
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_uitofp.mlir --args=1
   )
   set_tests_properties(UITOFP
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1"
   )
endif()