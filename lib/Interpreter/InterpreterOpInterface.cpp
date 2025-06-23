#include "mlir/Interpreter/InterpreterOpInterface.h"

#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

/// Include the definitions of the interpreter op interface.
#include "mlir/Interpreter/InterpreterOpInterface.cpp.inc"

namespace mlir {
EvalError::EvalError(llvm::StringRef message, bool shouldPrintStackTrace) : message(message.str()) {
    if ( shouldPrintStackTrace ) {
        llvm::raw_string_ostream stream(stacktrace);
        llvm::sys::PrintStackTrace(stream);
    }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const EvalValue& val) {
    // Special case i1 = bool
    auto valType = val.getType();
    if ( valType.isInteger(sizeof(bool)) ) {
        auto data = val.getData<bool>();
        if ( !data.empty() ) {
            os << data[0];
        }
    }
    // Handle index types
    else if ( valType.isIndex() ) {
        auto data = val.getData<intptr_t>();
        if ( !data.empty() ) {
            os << data[0];
        }
    }
    // Handle integers
    else if ( valType.isIntOrIndex() ) {
        auto data = val.getData<APInt>();
        if ( !data.empty() ) {
            os << data[0];
        }
    }
    // Handle float types
    else if ( mlir::isa<mlir::FloatType>(valType) ) {
        auto data = val.getData<APFloat>();
        if ( !data.empty() ) {
            data[0].print(os);
        }
    }
    // Handle pointer types
    else if ( mlir::isa<mlir::LLVM::LLVMPointerType>(valType) ) {
        auto data = val.getData<uint64_t>();
        if ( !data.empty() ) {
            os << "0x";
            os.write_hex(data[0]);
        }
    }
    // Default fallback - show raw bytes in hex
    else {
        const char* rawData  = val.getRawData();
        size_t      dataSize = val.getRawDataSizeInBytes();

        os << "0x";
        for ( size_t i = 0; i < dataSize; i++ ) {
            // Format each byte as hex
            char buffer[3];
            snprintf(buffer, sizeof(buffer), "%02x", (unsigned char) rawData[i]);
            os << buffer;
        }
    }
    return os;
}
}  // namespace mlir
