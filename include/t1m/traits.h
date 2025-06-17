#pragma once

#include <complex>
#include <type_traits>

namespace t1m {
template <typename T>
concept TensorScalarArithmetic =
    std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept TensorScalarCompound = std::is_same_v<T, std::complex<float>> ||
                               std::is_same_v<T, std::complex<double>>;
template <typename T>
concept TensorScalar = TensorScalarArithmetic<T> || TensorScalarCompound<T>;
}  // namespace t1m