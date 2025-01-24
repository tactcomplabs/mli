if(ENABLE_TESTING)
   add_test(NAME INTR.SMAX
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_smax.mlir --args=1,2
   )
   set_tests_properties(INTR.SMAX
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "2"
   )
endif()