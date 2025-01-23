if(ENABLE_TESTING)
   add_test(NAME FPEXT
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_fpext.mlir --args=1.5
   )
   set_tests_properties(FPEXT
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1.5"
   )
endif()