# ===- CMakeLists.txt - MLI (MLIR Interpreter)
#
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
cmake_minimum_required(VERSION 3.22)
project(MLI VERSION 1.0.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

# Set default build type to RelWithDebInfo if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
      STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Include directories
include_directories(BEFORE
  # Always prioritize the copies of files that are inside this
  # repo over ones found from previous installations
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Enable RTTI if needed
set(LLVM_ENABLE_RTTI ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Add subdirectories
add_subdirectory(lib)
add_subdirectory(src)

# Install include files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.td" PATTERN "*.inc")

# Install and export the targets
install(EXPORT MLITargets
        FILE MLITargets.cmake
        NAMESPACE MLI::
        DESTINATION lib/cmake/MLI)

# Generate and install package config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MLIConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MLIConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/MLIConfig.cmake"
    INSTALL_DESTINATION lib/cmake/MLI)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MLIConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MLIConfigVersion.cmake"
    DESTINATION lib/cmake/MLI)
    
option(ENABLE_TESTING "Enable Testing" OFF)
option(ENABLE_POLYGEIST "Enable Polygeist" OFF)
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
  add_executable(memManager test/mem_manager/testMemory.cpp)
  if (ENABLE_POLYGEIST)
    find_file(CGEIST cgeist PATHS $ENV{PATH})
    if (CGEIST)
        message(STATUS "Polygeist tests enabled, cgeist found at ${CGEIST}")
    else()
        message(WARNING "cgeist not found in \$PATH, Polygeist tests disabled")
    endif()
  endif()
endif()
