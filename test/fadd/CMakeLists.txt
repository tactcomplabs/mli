if(ENABLE_TESTING)
   add_test(NAME FADD
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_fadd.mlir --args=1.5,2.25
   )
   set_tests_properties(FADD
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 3.75"
   )
endif()