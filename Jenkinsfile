pipeline {
    agent {
        node {
            label 'nosferatu' 
        }
    } 

    environment {
        LLVM_DIR = "$HOME/.local/opt/llvm-polygeist"
    }

    stages {
        stage('Check Prerequisites') {
            steps {
                sh '''
                    if [[ ! -d "${LLVM_DIR}" ]]; then
                        echo "ERROR: LLVM not found at $LLVM_DIR"
                        exit 1
                    fi
                    source /etc/qlustar/common/skel/bash/bashrc
                    module load ninja/1.11.1-gcc-13.2.0-w72ajol
                '''
            }
        }
        stage('Build MLI') {
            steps {
                sh '''
                mkdir -p build
                cd build || exit
                rm -rf *

                cmake -G Ninja \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_PREFIX_PATH="$LLVM_DIR/lib/cmake" \
                    -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
                    -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir" \
                    -DCMAKE_C_COMPILER="$LLVM_DIR/bin/clang" \
                    -DCMAKE_CXX_COMPILER="$LLVM_DIR/bin/clang++" \
                    -DENABLE_TESTING=ON \
                ..
                ninja
                '''
            }
        }
        stage('Test') {
            steps {
                sh 'cd build && ctest'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'build/**/*.log', allowEmptyArchive: true
        }
    }
}