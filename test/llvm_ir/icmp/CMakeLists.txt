if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_icmp.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_icmp.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME ICMP
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_icmp.mlir --args=1,1)
    add_test(
        NAME ICMP_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_icmp.bc --args=1,1)
    set_tests_properties(ICMP ICMP_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION "1")
endif()