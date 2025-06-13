//===- MLIFormat.h - Multi-Level Interpreter Formatting Utils --*- C++ -*-===//
//
// Copyright (C) 2017-2024 Tactical Computing Laboratories, LLC
// All Rights Reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef MLI_FORMAT_H
#define MLI_FORMAT_H

#include <iostream>
#include <string>
#include <utility>
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"

namespace mli {
namespace fmt {

// "Specialize" std::to_string for common types that don't natively support it
template <typename T>
inline std::string to_string(T& val, bool isSigned = true) {
    std::string output;
    using stripped_type = std::remove_cv_t<T>;
    if constexpr(std::is_same_v<stripped_type, llvm::APInt>) {
        llvm::raw_string_ostream os(output);
        // The shovel operator for APInt automatically assumes the integer is signed
        // Use print instead to account for potential unsignedness
        val.print(os, isSigned);
        return os.str();
    }
    else if constexpr(std::is_same_v<stripped_type, llvm::APSInt> || std::is_same_v<stripped_type, llvm::APFloat>) {
        llvm::raw_string_ostream os(output);
        os << val;
        return os.str();
    }
    else {
        return std::to_string(val);
    }
}

// Specialization for std::pair
// We could add this to the above, but that would require more trickery for the type comparison
template <typename T1, typename T2>
inline std::string to_string(const std::pair<T1, T2>& p, bool isSigned = true) {
    return "(" + to_string(p.first, isSigned) + ", " + to_string(p.second, isSigned) + ")";
}

inline bool usePrettyPrint = true;

// ANSI escape codes for colors
inline const char* const RED     = "\033[31m";
inline const char* const GREEN   = "\033[32m";
inline const char* const YELLOW  = "\033[33m";
inline const char* const BLUE    = "\033[34m";
inline const char* const MAGENTA = "\033[35m";
inline const char* const CYAN    = "\033[36m";
inline const char* const WHITE   = "\033[37m";

// Text formatting
inline const char* const BOLD      = "\033[1m";
inline const char* const DIM       = "\033[2m";
inline const char* const ITALIC    = "\033[3m";
inline const char* const UNDERLINE = "\033[4m";
inline const char* const RESET     = "\033[0m";

// Utility functions for common formatting patterns
inline std::string error(const std::string& msg) {
    if (usePrettyPrint) {
        return std::string(BOLD) + RED + "error" + RESET + ": " + msg;
    }
    return "error: " + msg;
}

inline std::string warning(const std::string& msg) {
    if (usePrettyPrint) {
        return std::string(BOLD) + YELLOW + "warning" + RESET + ": " + msg;
    }
    return "warning: " + msg;
}

inline std::string success(const std::string& msg) {
    if (usePrettyPrint) {
        return std::string(BOLD) + GREEN + "success" + RESET + ": " + msg;
    }
    return "success: " + msg;
}

inline std::string info(const std::string& msg) {
    if (usePrettyPrint) {
        return std::string(BOLD) + BLUE + "info" + RESET + ": " + msg;
    }
    return "info: " + msg;
}

inline std::string highlight(const std::string& msg) {
    if (usePrettyPrint) {
        return std::string(BOLD) + CYAN + msg + RESET;
    }
    return msg;
}

inline std::string dim(const std::string& msg) {
    if (usePrettyPrint) {
        return std::string(DIM) + msg + RESET;
    }
    return msg;
}

inline std::string type(const std::string& msg) {
    if (usePrettyPrint) {
        return std::string(MAGENTA) + msg + RESET;
    }
    return msg;
}

// Helper functions for operation output formatting
inline void printOpName(llvm::raw_ostream &os, const std::string &opName) {
  os << info("Interpreting " + opName) << "\n";
}

template<typename T>
inline void printOperand(llvm::raw_ostream &os, const std::string &name, const T &value, bool isSigned = true) {
  os << dim(name + ": ") << to_string(value, isSigned) << "\n";
}

template<typename T>
inline void printResult(llvm::raw_ostream &os, const T &result, bool isSigned = true) {
  os << dim("result: ") << to_string(result, isSigned) << "\n";
}

} // namespace fmt
} // namespace mli

#endif // MLI_FORMAT_H
