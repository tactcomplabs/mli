if(ENABLE_TESTING)
   add_test(NAME INTR.UMIN
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_umin.mlir --args=1,2
   )
   set_tests_properties(INTR.UMIN
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1"
   )
endif()