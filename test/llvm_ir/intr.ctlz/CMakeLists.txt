if(ENABLE_TESTING)
   add_test(NAME INTR.CTLZ
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_ctlz.mlir --args=1
   )
   set_tests_properties(INTR.CTLZ
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "31"
   )
endif()