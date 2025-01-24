if(ENABLE_TESTING)
   add_test(NAME INTR.POW
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_pow.mlir --args=2.25,0.5
   )
   set_tests_properties(INTR.POW
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1.5"
   )
endif()