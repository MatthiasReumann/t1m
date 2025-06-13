#include <blis.h>
#include <gtest/gtest.h>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>
#include "t1m/bli/mappings.h"
#include "t1m/t1m.h"
#include "t1m/tensor.h"

namespace {

template <typename T> struct Alias {
  using value_type = T;
  using blis_type = T;
};

template <> struct Alias<std::complex<float>> {
  using value_type = float;
  using blis_type = scomplex;
};

template <> struct Alias<std::complex<double>> {
  using value_type = double;
  using blis_type = dcomplex;
};

template <std::size_t ndim>
std::size_t multiply(const std::array<std::size_t, ndim>& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<std::size_t>());
}

template <class T, std::size_t ndim_a, std::size_t ndim_b, std::size_t ndim_c>
void gemm_like(const std::string& labels_a, const std::string& labels_b,
               const std::string& labels_c, const std::size_t exp_m,
               const std::size_t exp_n, const std::size_t exp_k) {

  // BLIS requires scomplex and dcomplex types for their operations.
  // Hence, use this trick to convert the respective cpp standard type
  // to BLIS' custom types.
  using BLI_T = Alias<T>::blis_type;

  // For floating point numbers, U = T.
  // For complex numbers, U = std::complex<U>.
  using U = Alias<T>::value_type;

  std::allocator<BLI_T> alloc{};

  constexpr std::size_t max_dim = 20;
  constexpr U eps = std::numeric_limits<U>::epsilon();
  constexpr t1m::memory_layout layout = t1m::col_major;

  const BLI_T one(1);
  const BLI_T zero(0);
  const BLI_T minus_one(-1);

  for (std::size_t d = 2; d <= max_dim; ++d) {
    std::array<std::size_t, ndim_a> dims_a{};
    std::array<std::size_t, ndim_b> dims_b{};
    std::array<std::size_t, ndim_c> dims_c{};
    dims_a.fill(d);
    dims_b.fill(d);
    dims_c.fill(d);

    const std::size_t nelems_a = multiply(dims_a);
    const std::size_t nelems_b = multiply(dims_b);
    const std::size_t nelems_c = multiply(dims_c);

    const std::size_t m = std::pow(d, exp_m);
    const std::size_t n = std::pow(d, exp_n);
    const std::size_t k = std::pow(d, exp_k);

    BLI_T* data_a = alloc.allocate(nelems_a);
    BLI_T* data_b = alloc.allocate(nelems_b);
    BLI_T* data_c = alloc.allocate(nelems_c);
    BLI_T* data_ref = alloc.allocate(nelems_c);

    t1m::bli::randv<T>(nelems_a, data_a, 1);
    t1m::bli::randv<T>(nelems_b, data_b, 1);
    t1m::bli::setv<T>(BLIS_NO_CONJUGATE, nelems_c, &zero, data_c, 1);
    t1m::bli::setv<T>(BLIS_NO_CONJUGATE, nelems_c, &zero, data_ref, 1);

    t1m::tensor<T, ndim_a> a{dims_a, reinterpret_cast<T*>(data_a), layout};
    t1m::tensor<T, ndim_b> b{dims_b, reinterpret_cast<T*>(data_b), layout};
    t1m::tensor<T, ndim_c> c{dims_c, reinterpret_cast<T*>(data_c), layout};

    // Make sure to call the correct function for the respective types.
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      t1m::contract(T(1), a, labels_a, b, labels_b, T(0), c, labels_c);
    } else {
      t1m::contract(a, labels_a, b, labels_b, c, labels_c);
    }

    t1m::bli::gemm<T>(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &one,
                      data_a, 1, m, data_b, 1, k, &zero, data_ref, 1, m);
    t1m::bli::axpym<T>(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE, m,
                       n, &minus_one, data_c, 1, m, data_ref, 1, m);

    U norm_diff, a_norm, b_norm;
    t1m::bli::normv<T>(n, data_ref, 1, &norm_diff);
    t1m::bli::normv<T>(n, data_a, 1, &a_norm);
    t1m::bli::normv<T>(n, data_b, 1, &b_norm);

    const U estimator = norm_diff / (eps * d * a_norm * b_norm);

    EXPECT_LE(estimator, U(1));

    alloc.deallocate(data_ref, nelems_c);
    alloc.deallocate(data_c, nelems_c);
    alloc.deallocate(data_b, nelems_b);
    alloc.deallocate(data_a, nelems_a);
  }
}

};  // namespace

///-----------------------------------------------------------------------------
///                      \n Float \n
///-----------------------------------------------------------------------------

TEST(ContractionTest, FloatGemmLike) {
  gemm_like<float, 2, 2, 2>("ab", "bc", "ac", 1, 1, 1);
  gemm_like<float, 3, 3, 2>("abc", "bce", "ae", 1, 1, 2);
  gemm_like<float, 3, 2, 3>("abc", "cd", "abd", 2, 1, 1);
  gemm_like<float, 3, 3, 4>("abc", "cde", "abde", 2, 2, 1);
  gemm_like<float, 4, 3, 3>("abcd", "cde", "abe", 2, 1, 2);
  gemm_like<float, 4, 4, 4>("abcd", "cdef", "abef", 2, 2, 2);
}

///-----------------------------------------------------------------------------
///                      \n Double \n
///-----------------------------------------------------------------------------

TEST(ContractionTest, DoubleGemmLike) {
  gemm_like<float, 2, 2, 2>("ab", "bc", "ac", 1, 1, 1);
  gemm_like<float, 3, 3, 2>("abc", "bce", "ae", 1, 1, 2);
  gemm_like<double, 3, 2, 3>("abc", "cd", "abd", 2, 1, 1);
  gemm_like<double, 3, 3, 4>("abc", "cde", "abde", 2, 2, 1);
  gemm_like<double, 4, 3, 3>("abcd", "cde", "abe", 2, 1, 2);
  gemm_like<double, 4, 4, 4>("abcd", "cdef", "abef", 2, 2, 2);
}

///-----------------------------------------------------------------------------
///                      \n Complex (Float) \n
///-----------------------------------------------------------------------------

TEST(ContractionTest, ComplexFloatGemmLike) {
  gemm_like<std::complex<float>, 2, 2, 2>("ab", "bc", "ac", 1, 1, 1);
  gemm_like<std::complex<float>, 3, 3, 2>("abc", "bce", "ae", 1, 1, 2);
  gemm_like<std::complex<float>, 3, 2, 3>("abc", "cd", "abd", 2, 1, 1);
  gemm_like<std::complex<float>, 3, 3, 4>("abc", "cde", "abde", 2, 2, 1);
  gemm_like<std::complex<float>, 4, 3, 3>("abcd", "cde", "abe", 2, 1, 2);
  gemm_like<std::complex<float>, 4, 4, 4>("abcd", "cdef", "abef", 2, 2, 2);
}

///-----------------------------------------------------------------------------
///                      \n Complex (Double) \n
///-----------------------------------------------------------------------------

TEST(ContractionTest, ComplexDoubleGemmLike) {
  gemm_like<std::complex<double>, 2, 2, 2>("ab", "bc", "ac", 1, 1, 1);
  gemm_like<std::complex<double>, 3, 3, 2>("abc", "bce", "ae", 1, 1, 2);
  gemm_like<std::complex<double>, 3, 2, 3>("abc", "cd", "abd", 2, 1, 1);
  gemm_like<std::complex<double>, 3, 3, 4>("abc", "cde", "abde", 2, 2, 1);
  gemm_like<std::complex<double>, 4, 3, 3>("abcd", "cde", "abe", 2, 1, 2);
  gemm_like<std::complex<double>, 4, 4, 4>("abcd", "cdef", "abef", 2, 2, 2);
}