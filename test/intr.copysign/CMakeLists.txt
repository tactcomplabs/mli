if(ENABLE_TESTING)
   add_test(NAME INTR.COPYSIGN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_copysign.mlir --args=1.5,-2.25
   )
   set_tests_properties(INTR.COPYSIGN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: -1.5"
   )
endif()