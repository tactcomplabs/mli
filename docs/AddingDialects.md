# Adding a New Dialect to the MLI

This guide walks through the process of adding support for a new dialect to the MLIR Interpreter (MLI), using the Arith dialect as an example. How exciting! 🎉

## Overview

Adding a new dialect requires modifications to several parts of the codebase:

1. CMake configuration files
2. Main interpreter source files
3. Directory structure for the new dialect
4. Registration of dialect and interpreter implementations

## Step-by-Step Process

### 1. Update CMake Configuration

Update the following CMake files to include the new dialect:

#### a. `include/mlir/CMakeLists.txt`

```cmake
target_link_libraries(MLIRInterpreter
  PRIVATE
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect  # Add the new dialect
  MLIRLLVMDialect
  MLIRSupport
)
```

#### b. `lib/CMakeLists.txt`

- Add the new source file:

```cmake
add_library(MLIRInterpreterLib SHARED
  Interpreter/InterpreterOpInterface.cpp
  Interpreter/Dialects/Func/FuncInterpreter.cpp
  Interpreter/Dialects/LLVM/LLVMInterpreter.cpp
  Interpreter/Dialects/Arith/ArithInterpreter.cpp  # Add new interpreter file
)
```

- Update target linkage:

```cmake
target_link_libraries(MLIRInterpreterLib
  PUBLIC
  MLIRIR
  MLIRArithDialect  # Add new dialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRSupport
)
```

#### c. `lib/Interpreter/Dialects/CMakeLists.txt`

```cmake
add_subdirectory(Func)
add_subdirectory(LLVM)
add_subdirectory(Arith)  # Add new dialect directory
```

#### d. `lib/Interpreter/Dialects/Arith/CMakeLists.txt` (new file)

```cmake
add_mlir_library(MLIRArithInterpreter
  ArithInterpreter.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interpreter

  DEPENDS
  MLIRInterpreterOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRInterpreter
  MLIRArithDialect
  MLIRSupport
)
```

#### e. `src/CMakeLists.txt`

```cmake
target_link_libraries(mli
  PRIVATE
  MLIRInterpreterLib
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRArithDialect  # Add new dialect
  MLIRSupport
  MLIRParser
  MLIRTransforms
)
```

### 2. Update Main Source Files

Modify `src/MLI.cpp` to:

1. Include the new dialect header:

```cpp
#include "mlir/Dialect/Arith/IR/Arith.h" // May vary between dialects
#include "mlir/Interpreter/Dialects/ArithInterpreter.h"
```

2. Register the dialect and interpreter:

```cpp
context.getOrLoadDialect<mlir::arith::ArithDialect>();
interpreter.registerDialectInterpreter<mlir::ArithInterpreter>();
```

### 3. Add Test Directory

Add a new test directory under `test/` for the dialect-specific tests:

```cmake
if (ENABLE_TESTING)
  add_subdirectory(bytecode)
  add_subdirectory(llvm_ir)
  add_subdirectory(mem_manager)
  add_subdirectory(arith)  # Add new test directory
endif()
```

## Directory Structure

The new dialect should follow this directory structure:

```
lib/Interpreter/Dialects/Arith/
├── ArithInterpreter.cpp
└── CMakeLists.txt

include/mlir/Interpreter/Dialects/
├── ArithInterpreter.h
├── FuncInterpreter.h
├── LLVMInterpreter.h
└── CMakeLists.txt

test/arith/
├── CMakeLists.txt
└── [test directories]
```

## Notes

- Follow the existing pattern for interpreter implementation
- Add appropriate tests in the new test directory
- Make sure all new files are properly included in version control

## Common Issues

- Missing library dependencies in CMake files
- Forgetting to register the dialect or interpreter in `MLI.cpp`
- Incorrect include paths for new headers
- Missing test directory configuration

Remember to run the test suite after adding the new dialect to ensure everything works as expected.
