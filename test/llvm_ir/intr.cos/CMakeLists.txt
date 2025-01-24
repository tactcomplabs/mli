if(ENABLE_TESTING)
   add_test(NAME INTR.COS
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_cos.mlir --args=0
   )
   set_tests_properties(INTR.COS
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "1"
   )
endif()