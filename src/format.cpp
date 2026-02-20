#include "quantcore/format.hpp"

namespace quantcore {

std::string_view format_version_string(FormatVersion version) {
    switch (version) {
        case FormatVersion::kV1:
            return "v1";
        default:
            return "unknown";
    }
}

}  // namespace quantcore
