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
