if(ENABLE_TESTING)
   add_test(NAME FPTOUI
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_fptoui.mlir --args=1.5
   )
   set_tests_properties(FPTOUI
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1"
   )
endif()