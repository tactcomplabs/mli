if(ENABLE_TESTING)
   add_test(NAME INTR.SMIN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_smin.mlir --args=1,2
   )
   set_tests_properties(INTR.SMIN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1"
   )
endif()