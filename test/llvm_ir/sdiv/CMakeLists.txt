if(ENABLE_TESTING)
   add_test(NAME SDIV
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_sdiv.mlir --args=1,2
   )
   set_tests_properties(SDIV
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "0"
   )
endif()