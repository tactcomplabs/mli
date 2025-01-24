if(ENABLE_TESTING)
   add_test(NAME INTR.EXP
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_exp.mlir --args=1
   )
   set_tests_properties(INTR.EXP
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "2.71828175"
   )
endif()