if (ENABLE_TESTING)
  add_executable(memManager testMemory.cpp)
  add_test(NAME MEM_MANAGER COMMAND memManager PASS_REGULAR_EXPRESSION "SUCCESS")
endif()
