if(ENABLE_TESTING)
   add_test(NAME BITCAST_INTTOFP
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_bitcast.mlir --args=1069547520 --func=itofp
   )
   set_tests_properties(BITCAST_INTTOFP
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1.5"
   )
   add_test(NAME BITCAST_FPTOINT
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_bitcast.mlir --args=1.5 --func=fptoi
   )
   set_tests_properties(BITCAST_FPTOINT
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "result: 1069547520"
   )
endif()
