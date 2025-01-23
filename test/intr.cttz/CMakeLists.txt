if(ENABLE_TESTING)
   add_test(NAME INTR.CTTZ
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_cttz.mlir --args=8
   )
   set_tests_properties(INTR.CTTZ
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 3"
   )
endif()