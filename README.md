# MLIR Interpreter (Multi-Level Interpreter)

Out-of-Tree Interpreter for executing MLIR code. Based on [work](https://discourse.llvm.org/t/rfc-mlir-interpreter-framework/63567) originally published by @lchien

# Building

**DEVELOPER NOTE**: If you end up seeing weird issues when making changes to this code be aware that it may be linking against the installed version of the headers. To fix this, make sure to remove the installed headers and re-run the build process

```bash
mkdir build && cd build
cmake -G Ninja -DLLVM_DIR=/opt/homebrew/opt/llvm@19/lib/cmake/llvm -DMLIR_DIR=/opt/homebrew/opt/llvm@19/lib/cmake/mlir -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_BUILD_TYPE=RelWDebInfo ../ -DCMAKE_INSTALL_PREFIX=/opt/homebrew/opt/llvm@19
```

# Testing

- `mli` executable takes in a `.mlir` file and runs the interpreter on it.

_Example_: `build/src/mli <path-to-mlir-file> --args=5,10`
