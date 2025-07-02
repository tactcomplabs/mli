add_mlir_library(MLIRCFInterpreter
  CFInterpreter.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interpreter

  DEPENDS
  MLIRInterpreterOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRInterpreter
  MLIRControlFlowDialect
  MLIRSupport
)
