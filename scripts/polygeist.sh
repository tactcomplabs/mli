#!/usr/bin/env bash
set -euo pipefail

#─── Defaults and CLI parsing ────────────────────────────────────────────────
PREFIX="/opt/llvm-polygeist"
BRANCH="main"
REPO_DIR="polygeist"

usage() {
  cat <<EOF
Usage: $0 [--prefix DIR] [--branch NAME] [--repo-dir DIR]
Clones and builds LLVM+Clang+MLIR+Polygeist into \$PREFIX.

Options:
  --prefix DIR     Install prefix (default: $PREFIX)
  --branch NAME    Polygeist Git branch (default: $BRANCH)
  --repo-dir DIR   Clone directory (default: $REPO_DIR)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
  --prefix)
    PREFIX="$2"
    shift 2
    ;;
  --branch)
    BRANCH="$2"
    shift 2
    ;;
  --repo-dir)
    REPO_DIR="$2"
    shift 2
    ;;
  -h | --help) usage ;;
  *)
    echo "Unknown option: $1"
    usage
    ;;
  esac
done

#─── Dependency Check ────────────────────────────────────────────────────────
for cmd in git cmake ninja; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: '$cmd' is required but not installed." >&2
    exit 2
  fi
done

#─── Clone and Prepare Sources ───────────────────────────────────────────────
cd ..
if [[ -d "$REPO_DIR" ]]; then
  echo "✔️  Reusing existing directory '$REPO_DIR'"
else
  echo "⏳ Cloning Polygeist into '$REPO_DIR'..."
  git clone --depth 1 --branch "$BRANCH" https://github.com/llvm/Polygeist.git "$REPO_DIR"
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
