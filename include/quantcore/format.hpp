#pragma once

#include <cstdint>
#include <string_view>

namespace quantcore {

enum class FormatVersion : std::uint32_t {
    kV1 = 1,
};

struct BinaryFormatSpec {
    static constexpr std::uint32_t kBitsPerElement = 1;
    static constexpr std::uint32_t kElementsPerBlock = 64;
    static constexpr bool kRowMajor = true;
    static constexpr bool kLsbFirst = true;
    static constexpr std::int8_t kBitValueZero = -1;
    static constexpr std::int8_t kBitValueOne = +1;
};

struct TernaryFormatSpec {
    static constexpr std::uint32_t kBitsPerElementPerPlane = 1;
    static constexpr std::uint32_t kBitPlanes = 2;
    static constexpr std::uint32_t kElementsPerBlock = 64;
    static constexpr bool kRowMajor = true;
    static constexpr bool kLsbFirst = true;
};

std::string_view format_version_string(FormatVersion version);

}  // namespace quantcore
