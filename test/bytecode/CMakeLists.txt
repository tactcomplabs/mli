if (ENABLE_TESTING)
  add_test(NAME BYTECODE 
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}	
	   COMMAND mli test-bytecode.bc --args=1,2
  )
endif()
