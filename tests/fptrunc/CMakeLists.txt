if(ENABLE_TESTING)
   add_test(NAME FPTRUNC
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_fptrunc.mlir --args=3.14159268
   )
   set_tests_properties(FPTRUNC
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 3.1415925"
   )
endif()