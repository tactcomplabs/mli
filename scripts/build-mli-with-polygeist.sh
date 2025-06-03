#!/bin/bash
#
# scripts/slurm/build-ci-mli-llvm18.1.3.sh
#
# Copyright (C) 2017-2025 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# This file is a part of the Rev package.  For license
# information, see the LICENSE file in the top level directory of
# this distribution.
#
#
# Sample SLURM batch script
#
# Usage: sbatch -N1 build.sh
#
# This command requests 1 nodes for execution
#

#-- Stage 1: load the necessary modules
set -eou pipefail

source /etc/qlustar/common/skel/bash/bashrc
module load ninja/1.11.1-gcc-13.2.0-w72ajol
export LLVM_DIR=$HOME/.local/opt/llvm-polygeist

if [[ ! -d $LLVM_DIR ]]; then
	echo "ERROR: LLVM does not exist at $LLVM_DIR"
	exit 1
fi

# exec >> "rev.jenkins.${SLURM_JOB_ID}.out" 2>&1

#-- Stage 2: setup the build directories
mkdir -p build
cd build || exit
rm -rf *

#-- Stage 3: initiate the build
           cmake -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH="$LLVM_DIR/lib/cmake" \
          -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
          -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir" \
          -DCMAKE_C_COMPILER="$LLVM_DIR/bin/clang" \
          -DCMAKE_CXX_COMPILER="$LLVM_DIR/bin/clang++" \
		  -DENABLE_TESTING=ON \
          ../
          ninja

#-- Stage 4: test everything
ctest

#-- EOF
