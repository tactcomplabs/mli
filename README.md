# MLIR Interpreter (Multi-Level Interpreter)

Out-of-Tree Interpreter for executing MLIR code. Based on [work](https://discourse.llvm.org/t/rfc-mlir-interpreter-framework/63567) originally published by @lchien

# Build Instructions (with LLVM already installed)

```bash
mkdir build && cd build
cmake -G Ninja -DLLVM_DIR=/opt/homebrew/opt/llvm@19/lib/cmake/llvm -DMLIR_DIR=/opt/homebrew/opt/llvm@19/lib/cmake/mlir -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_BUILD_TYPE=RelWDebInfo ../ -DCMAKE_INSTALL_PREFIX=/opt/homebrew/opt/llvm@19
```
If you plan on contributing to this repository, we suggest you run `git config core.hooksPath .githooks` after cloning the repository. This sets up the precommit hooks that automatically enforce coding standards.

## Building Polygeist
We provide support for [Polygeist](http://polygeist.llvm.org), a C/C++ frontend that can emit MLIR source. This is optional for building `mli`, but is helpful if the user wants to work from C++ source files. Polygeist can be built by:
```bash
cd scripts
./polygeist.sh
```
Note that this involves a clone and build of the full LLVM project, which takes a significant amount of time. See the script for more details about setting the install prefix directory and the commit used for LLVM

### MacOS Build Example

```bash
brew install cmake ninja llvm@19
cmake -G Ninja -DLLVM_DIR=/opt/homebrew/opt/llvm@19/lib/cmake/llvm -DMLIR_DIR=/opt/homebrew/opt/llvm@19/lib/cmake/mlir -DCMAKE_EXPORT_COMPILE_COMMANDS=On ../
```

# Testing

- `mli` executable takes in a `.mlir` file and runs the interpreter on it.

_Example_: `build/src/mli <path-to-mlir-file> --args=5,10`

# Adding New Dialects

See [Adding New Dialects](docs/AddingDialects.md)
