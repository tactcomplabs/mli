# ===- lib/CMakeLists.txt
#
# Copyright (C) 2017-2024 Tactical Computing Laboratories, LLC
# All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set CMake policies
cmake_policy(SET CMP0057 NEW)
cmake_policy(SET CMP0116 NEW)

add_library(MLIRInterpreterLib SHARED
  Interpreter/Interpreter.cpp
  Interpreter/InterpreterOpInterface.cpp
  Interpreter/Dialects/Func/FuncInterpreter.cpp
  Interpreter/Dialects/LLVM/LLVMInterpreter.cpp
)

# Include directories for the library
target_include_directories(MLIRInterpreterLib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# List of required LLVM components
set(LLVM_LINK_COMPONENTS
  Core
  Support
  Demangle
  Analysis
  MC
  BitReader
  BitWriter
  TransformUtils
  Target
  Remarks
)

# Map LLVM components to library names
llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_LINK_COMPONENTS})

# Link MLIRInterpreterLib against necessary libraries
target_link_libraries(MLIRInterpreterLib
  PUBLIC
  # MLIR libraries
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRSupport
  MLIRParser
  MLIRTransforms

  # LLVM libraries
  ${LLVM_LIBS}
)

# TableGen for InterpreterOpInterface
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/mlir/Interpreter/InterpreterOpInterface.td)

# Set the output directory for generated files
set(OUTPUT_INC_DIR ${CMAKE_BINARY_DIR}/include/mlir/Interpreter)
file(MAKE_DIRECTORY ${OUTPUT_INC_DIR})

# Generate the header file
add_custom_command(
  OUTPUT ${OUTPUT_INC_DIR}/InterpreterOpInterface.h.inc
  COMMAND mlir-tblgen -gen-op-interface-decls
          -I ${CMAKE_SOURCE_DIR}/include
          -I ${LLVM_INCLUDE_DIRS}
          -I ${MLIR_INCLUDE_DIRS}
          ${LLVM_TARGET_DEFINITIONS}
          -o ${OUTPUT_INC_DIR}/InterpreterOpInterface.h.inc
  DEPENDS ${LLVM_TARGET_DEFINITIONS}
  COMMENT "Generating InterpreterOpInterface.h.inc"
)

# Generate the source file
add_custom_command(
  OUTPUT ${OUTPUT_INC_DIR}/InterpreterOpInterface.cpp.inc
  COMMAND mlir-tblgen -gen-op-interface-defs
          -I ${CMAKE_SOURCE_DIR}/include
          -I ${LLVM_INCLUDE_DIRS}
          -I ${MLIR_INCLUDE_DIRS}
          ${LLVM_TARGET_DEFINITIONS}
          -o ${OUTPUT_INC_DIR}/InterpreterOpInterface.cpp.inc
  DEPENDS ${LLVM_TARGET_DEFINITIONS}
  COMMENT "Generating InterpreterOpInterface.cpp.inc"
)

# Create a custom target for the generated files
add_custom_target(MLIRInterpreterOpInterfaceIncGen
  DEPENDS
    ${OUTPUT_INC_DIR}/InterpreterOpInterface.h.inc
    ${OUTPUT_INC_DIR}/InterpreterOpInterface.cpp.inc
)

# Ensure the generated files are available before building the library
add_dependencies(MLIRInterpreterLib MLIRInterpreterOpInterfaceIncGen)

# Install the library target
install(TARGETS MLIRInterpreterLib
  EXPORT MLITargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install public headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.td" PATTERN "*.inc"
)

# Install generated headers
install(FILES
  ${OUTPUT_INC_DIR}/InterpreterOpInterface.h.inc
  ${OUTPUT_INC_DIR}/InterpreterOpInterface.cpp.inc
  DESTINATION include/mlir/Interpreter
)
