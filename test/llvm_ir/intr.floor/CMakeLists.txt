if(ENABLE_TESTING)
    execute_process(COMMAND mlir-opt --emit-bytecode -o test_intr_floor.bc ${CMAKE_CURRENT_SOURCE_DIR}/test_intr_floor.mlir
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME INTR_FLOOR
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND mli test_intr_floor.mlir --args=1.5)
    add_test(
        NAME INTR_FLOOR_BYTECODE
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND mli test_intr_floor.bc --args=1.5)
    set_tests_properties(INTR_FLOOR INTR_FLOOR_BYTECODE PROPERTIES TIMEOUT 60 LABELS "all"
                                                    PASS_REGULAR_EXPRESSION ".+result: .+1")
endif()