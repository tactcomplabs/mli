if(ENABLE_TESTING)
   add_test(NAME INTR.EXP2
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_exp2.mlir --args=3
   )
   set_tests_properties(INTR.EXP2
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "8"
   )
endif()