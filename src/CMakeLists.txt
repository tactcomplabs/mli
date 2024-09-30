# ===- src/CMakeLists.txt
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
add_executable(MLI MLI.cpp)

target_include_directories(interpreter-driver PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
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

# Link interpreter-driver against necessary libraries
target_link_libraries(interpreter-driver
  PRIVATE
  MLIRInterpreterLib
  # MLIR libraries
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRSupport
  MLIRParser
  MLIRTransforms
  LLVMSupport

  # LLVM libraries
  ${LLVM_LIBS}
)
