if(ENABLE_TESTING)
   add_test(NAME INTR.BSWAP
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_intr_bswap.mlir --args=100000
   )
   set_tests_properties(INTR.BSWAP
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 2693136640"
   )
endif()