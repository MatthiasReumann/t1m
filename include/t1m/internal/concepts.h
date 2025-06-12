#pragma once

#include <complex>
#include <type_traits>

namespace t1m {
namespace internal {
template <typename T>
concept Real = (std::is_same_v<T, float> || std::is_same_v<T, double>);

template <typename T>
concept Complex = (std::is_same_v<T, std::complex<float>> ||
                   std::is_same_v<T, std::complex<double>>);
}  // namespace internal
}  // namespace t1m