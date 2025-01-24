if(ENABLE_TESTING)
   add_test(NAME INTR.BITREVERSE
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_bitreverse.mlir --args=3
   )
   set_tests_properties(INTR.BITREVERSE
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "3221225472"
   )
endif()