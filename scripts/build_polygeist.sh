#!/usr/bin/env bash
set -euo pipefail

source /etc/qlustar/common/skel/bash/bashrc
module load ninja/1.11.1-gcc-13.2.0-w72ajol

#─── Defaults and CLI parsing ────────────────────────────────────────────────
PREFIX="$HOME/.local/opt/llvm-polygeist"
BRANCH="main"
REPO_DIR="polygeist"

#─── Clone and Prepare Sources ───────────────────────────────────────────────
cd ..
if [[ -d "$REPO_DIR" ]]; then
  echo "✔️  Reusing existing directory '$REPO_DIR'"
else
  echo "⏳ Cloning Polygeist into '$REPO_DIR'..."
  git clone --depth 1 --shallow-submodules --branch "$BRANCH" https://github.com/llvm/Polygeist.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "⏳ Initializing submodules (LLVM monorepo)..."
git submodule update --init --recursive

#─── Configure & Build ───────────────────────────────────────────────────────
mkdir -p build && cd build
echo "⏳ Configuring with CMake → Ninja generator..."
cmake -G Ninja "../llvm-project/llvm" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=".." \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"

echo "⏳ Building (this may take 20–30 minutes)..."
ninja

echo "⏳ Installing into '$PREFIX' (you may need sudo)..."
ninja install

#─── Post-Install README ────────────────────────────────────────────────────
cat <<EOF >"$PREFIX/README"
This directory contains LLVM + Clang + MLIR + Polygeist installed by build script.
Binaries live in:  $PREFIX/bin
Libraries  live in:  $PREFIX/lib
EOF

#─── Usage Instructions ───────────────────────────────────────────────────────
cat <<EOF

✅ Installation complete!

To use these tools, add the following to your shell rc (e.g. ~/.bashrc or ~/.zshrc):

  export PATH="$PREFIX/bin:\$PATH"
  export LD_LIBRARY_PATH="$PREFIX/lib:\$LD_LIBRARY_PATH"

You’re all set! 🎉
EOF
