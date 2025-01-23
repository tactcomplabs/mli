if(ENABLE_TESTING)
   add_test(NAME INTR.LOG2
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_log2.mlir --args=32
   )
   set_tests_properties(INTR.LOG2
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 5"
   )
endif()