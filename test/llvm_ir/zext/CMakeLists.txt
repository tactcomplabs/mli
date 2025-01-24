if(ENABLE_TESTING)
   add_test(NAME ZEXT
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_zext.mlir --args=-1
   )
   set_tests_properties(ZEXT
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "4294967295"
   )
endif()