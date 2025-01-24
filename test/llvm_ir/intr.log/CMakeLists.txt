if(ENABLE_TESTING)
   add_test(NAME INTR.LOG
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_log.mlir --args=1.5
   )
   set_tests_properties(INTR.LOG
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "0.405465096"
   )
endif()