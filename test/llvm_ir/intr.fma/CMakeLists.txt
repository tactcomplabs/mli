if(ENABLE_TESTING)
   add_test(NAME INTR.FMA
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_fma.mlir --args=1.5,2.25,3.125
   )
   set_tests_properties(INTR.FMA
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "6.5"
   )
endif()