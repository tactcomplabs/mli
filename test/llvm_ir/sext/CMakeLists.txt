if(ENABLE_TESTING)
   add_test(NAME SEXT
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_sext.mlir --args=1
   )
   set_tests_properties(SEXT
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1"
   )
endif()