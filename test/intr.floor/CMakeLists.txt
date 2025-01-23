if(ENABLE_TESTING)
   add_test(NAME INTR.FLOOR
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_floor.mlir --args=1.5
   )
   set_tests_properties(INTR.FLOOR
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1"
   )
endif()