if(ENABLE_TESTING)
   add_test(NAME MLIR_CONSTANT_INT
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_mlir_constant.mlir  --func=int
   )
   set_tests_properties(MLIR_CONSTANT_INT
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "Initializing constant 42"  
)
  add_test(NAME MLIR_CONSTANT_FLOAT
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mli test_mlir_constant.mlir  --func=float
   )
   set_tests_properties(MLIR_CONSTANT_FLOAT
      PROPERTIES
      TIMEOUT 60
      LABELS "all"
      PASS_REGULAR_EXPRESSION "Initializing constant 2.5"
   )
endif()
