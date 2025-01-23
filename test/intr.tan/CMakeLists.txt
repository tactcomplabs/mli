if(ENABLE_TESTING)
   add_test(NAME INTR.TAN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_tan.mlir --args=1.5
   )
   set_tests_properties(INTR.TAN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 14.10141994717172"
   )
endif()