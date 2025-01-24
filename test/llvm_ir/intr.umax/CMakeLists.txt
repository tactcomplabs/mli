if(ENABLE_TESTING)
   add_test(NAME INTR.UMAX
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_umax.mlir --args=1,2
   )
   set_tests_properties(INTR.UMAX
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "2"
   )
endif()