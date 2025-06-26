function(add_mli_test)
  set(options "")
  set(oneValueArgs name_base subdir mli_args regex filename_base label function_name)
  set(multiValueArgs "")
  cmake_parse_arguments(test  # Lowercase prefix
                        "${options}"
                        "${oneValueArgs}"
                        "${multiValueArgs}"
                        ${ARGN})

  # Use the prefixed lowercase variables
  if(NOT test_name_base OR NOT test_subdir OR NOT test_regex)
    message(FATAL_ERROR "add_mli_test requires name_base, subdir, "
                        "and regex arguments.")
  endif()

  # TODO: Maybe add parsing of labels in the future
  if(NOT test_label)
    set(test_label "all")
  endif()

  if(test_function_name)
    string(TOUPPER ${test_function_name} function_upper)
    set(mlir_test_name ${test_name_base}_${function_upper})
    set(bc_test_name ${test_name_base}_${function_upper}_BYTECODE)
  else()
    set(mlir_test_name ${test_name_base})
    set(bc_test_name ${test_name_base}_BYTECODE)
  endif()

  # Strip extraneous spaces between args
  if(test_mli_args)
    string(REPLACE ", " "," test_mli_args ${test_mli_args})
  endif()

  # --- Determine file paths ---
  if(NOT test_filename_base)
    # Convert the test base name to lowercase for consistent processing.
    string(TOLOWER ${test_subdir} test_filename_base_lower)

    get_filename_component(current_dir_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    string(TOLOWER ${current_dir_name} dir_prefix)
    set(prefix_to_remove "${dir_prefix}_") # Creates "arith_", "llvm_ir_", etc.

    string(REPLACE "${prefix_to_remove}" "" test_filename_base_lower ${test_filename_base_lower})
    string(REPLACE "." "_" test_filename_base_lower ${test_filename_base_lower})

    set(test_filename_base "test_${test_filename_base_lower}")
  endif()

  set(mlir_source_rel_path ${test_subdir}/${test_filename_base}.mlir)
  set(mlir_source_abs_path ${CMAKE_CURRENT_SOURCE_DIR}/${mlir_source_rel_path})
  set(bc_file_abs_path ${CMAKE_CURRENT_BINARY_DIR}/${test_subdir}/${test_filename_base}.bc)

  # Relative to CMAKE_CURRENT_BINARY_DIR
  set(bc_file_rel_path ${test_subdir}/${test_filename_base}.bc)

  # Create a unique prefix for the target name based on the calling directory (Hopefully this won't break)
  file(RELATIVE_PATH rel_path_from_root ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "/" "_" sanitized_path_prefix ${rel_path_from_root})

  # Handle cases where CMAKE_CURRENT_SOURCE_DIR might be CMAKE_SOURCE_DIR,
  # resulting in an empty or "." relative path.
  if(NOT sanitized_path_prefix OR sanitized_path_prefix STREQUAL ".")
    if(sanitized_path_prefix STREQUAL ".")
      # Request that the operation's test be moved to its constituent dialect subdirectory
      set(sanitized_path_prefix "move_this_test_to_its_own_dialect_directory_like_llvm_ir_or_arith")
    endif()
  endif()

  set(gen_target_name Generate_${sanitized_path_prefix}_${test_name_base}_${test_function_name}_BC)

  # --- Ensure source file exists ---
  if(NOT EXISTS ${mlir_source_abs_path})
    message(FATAL_ERROR "MLIR source file not found for test ${test_name_base}: "
                        "${mlir_source_abs_path}")
  endif()

  # --- Ensure mlir-opt executable can be found from imported target ---
  get_target_property(MLIR_MLIR_OPT_EXECUTABLE mlir-opt LOCATION)
  if(NOT MLIR_MLIR_OPT_EXECUTABLE)
    message(FATAL_ERROR "mlir-opt not found; cannot emit bytecode")
  endif()
  set(mlir_tool ${MLIR_MLIR_OPT_EXECUTABLE})
  set(mlir_tool_args --emit-bytecode)

  # --- Define how to generate the .bc file (runs during build) ---
  add_custom_command(
    OUTPUT ${bc_file_abs_path}
    COMMAND ${mlir_tool} ${mlir_tool_args}
            -o ${bc_file_abs_path} ${mlir_source_abs_path}
    DEPENDS ${mlir_source_abs_path}
    COMMENT "Generating ${bc_file_rel_path}"
    VERBATIM
  )

  # --- Create a target that triggers the generation ---
  add_custom_target(${gen_target_name} ALL DEPENDS ${bc_file_abs_path})

  # Add support for calling specific functions though much preferred to just
  # keep it to main
  set(mli_function_arg "")
  if(test_function_name)
    set(mli_function_arg "--func=${test_function_name}")
  endif()

  # Disable pretty printing for tests
  # This removes invisible characters that can interfere with regex pattern matching
  set(mli_formatting_args "--pretty-print=false")

  # --- Add the .mlir test ---
  add_test(
    NAME ${mlir_test_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${test_subdir}
    COMMAND $<TARGET_FILE:mli> ${test_filename_base}.mlir ${mli_function_arg} ${test_mli_args} ${mli_formatting_args}
  )

  set_tests_properties(${mlir_test_name} PROPERTIES
    TIMEOUT 60
    LABELS "${test_label}"
    PASS_REGULAR_EXPRESSION ${test_regex}
  )

  # --- Add the .bc test ---
  add_test(
    NAME ${bc_test_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_subdir}
    COMMAND $<TARGET_FILE:mli> ${test_filename_base}.bc ${mli_function_arg} ${test_mli_args} ${mli_formatting_args}
  )
  set_tests_properties(${bc_test_name} PROPERTIES
    TIMEOUT 60
    LABELS "${test_label}"
    PASS_REGULAR_EXPRESSION "${test_regex}"
  )
endfunction()

if(ENABLE_TESTING)
  add_subdirectory(llvm_ir)
  add_subdirectory(arith)
  add_subdirectory(func)
  #add_subdirectory(scf)
  add_subdirectory(mem_manager)
  if(ENABLE_POLYGEIST)
    add_subdirectory(polygeist)
  endif()
endif()
