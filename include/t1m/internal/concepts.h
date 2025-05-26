#pragma once

#include <type_traits>

namespace t1m {
namespace internal {
template <typename T>
concept Real =
    (std::is_same_v<T, float> || std::is_same_v<T, double>);
}
}  // namespace t1m