# MLIR Interpreter (Multi-Level Interpreter)

Out-of-Tree Interpreter for executing MLIR code.

# Building

```bash
mkdir build && cd build
cmake -G Ninja -DLLVM_DIR=/path/to/llvm/install/prefix -DMLIR_DIR=/path/to/mlir/install/prefix -DCMAKE_EXPORT_COMPILE_COMMANDS=On ../
```

# Testing

- `mli` executable takes in a `.mlir` file and runs the interpreter on it.

*Example*: `build/src/mli <path-to-mlir-file> --args=5,10`
