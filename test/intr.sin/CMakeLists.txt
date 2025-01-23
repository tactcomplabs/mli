if(ENABLE_TESTING)
   add_test(NAME INTR.SIN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_sin.mlir --args=0
   )
   set_tests_properties(INTR.SIN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 0"
   )
endif()