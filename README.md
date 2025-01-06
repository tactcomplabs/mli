# MLIR Interpreter (Multi-Level Interpreter)

Out-of-Tree Interpreter for executing MLIR code. Based on [work](https://discourse.llvm.org/t/rfc-mlir-interpreter-framework/63567) originally published by @lchien

# Building
**DEVELOPER NOTE**: I believe at this point it is important to specify the `-DCMAKE_INSTALL_PREFIX` to be something other than the LLVM install location because something in the
CMake will end up compiling against the installed version of the headers instead of the ones included in this repo. Not sure how to fix but if anyone smarter comes around 
feel free to make a PR :) 

```bash
mkdir build && cd build
cmake -G Ninja -DLLVM_DIR=/path/to/llvm/install/prefix -DMLIR_DIR=/path/to/mlir/install/prefix -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_INSTALL_PREFIX=$HOME/.local/opt ../
```

# Testing

- `mli` executable takes in a `.mlir` file and runs the interpreter on it.

*Example*: `build/src/mli <path-to-mlir-file> --args=5,10`
