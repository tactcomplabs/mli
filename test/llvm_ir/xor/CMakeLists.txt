if(ENABLE_TESTING)
   add_test(NAME XOR
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_xor.mlir --args=1,2
   )
   set_tests_properties(XOR
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "3"
   )
endif()