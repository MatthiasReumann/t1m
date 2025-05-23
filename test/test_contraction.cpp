#include <blis/blis.h>
#include <gtest/gtest.h>
#include <functional>
#include <limits>
#include <numeric>
#include "t1m/internal/blis.h"
#include "t1m/internal/t1m.h"
#include "t1m/internal/tensor.h"

namespace {
template <std::size_t ndim>
std::size_t multiply(const std::array<std::size_t, ndim>& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<std::size_t>());
}

template <typename T, std::size_t ndim_a, std::size_t ndim_b,
          std::size_t ndim_c>
void gemm_like(const std::string& labels_a, const std::string& labels_b,
               const std::string& labels_c, const std::size_t exp_m,
               const std::size_t exp_n, const std::size_t exp_k) {
  std::allocator<T> alloc{};

  constexpr std::size_t max_dim = 20;
  constexpr T eps = std::numeric_limits<T>::epsilon();
  constexpr t1m::memory_layout layout = t1m::col_major;

  const T one = T(1);
  const T zero = T(0);
  const T mone = T(-1);

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

    T* data_a = alloc.allocate(nelems_a);
    T* data_b = alloc.allocate(nelems_b);
    T* data_c = alloc.allocate(nelems_c);
    T* data_ref = alloc.allocate(nelems_c);

    t1m::bli::randv<T>(nelems_a, data_a, 1);
    t1m::bli::randv<T>(nelems_b, data_b, 1);
    t1m::bli::setv<T>(BLIS_NO_CONJUGATE, nelems_c, &zero, data_c, 1);
    t1m::bli::setv<T>(BLIS_NO_CONJUGATE, nelems_c, &zero, data_ref, 1);

    t1m::tensor<T, ndim_a> a{dims_a, data_a, layout};
    t1m::tensor<T, ndim_b> b{dims_b, data_b, layout};
    t1m::tensor<T, ndim_c> c{dims_c, data_c, layout};

    t1m::contract(T(1), a, labels_a, b, labels_b, T(0), c, labels_c);

    t1m::bli::gemm<T>(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &one,
                      data_a, 1, m, data_b, 1, k, &zero, data_ref, 1, m);
    t1m::bli::axpym<T>(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE, m,
                       n, &mone, data_c, 1, m, data_ref, 1, m);

    T norm_diff, a_norm, b_norm;
    t1m::bli::normv<T>(n, data_ref, 1, &norm_diff);
    t1m::bli::normv<T>(n, data_a, 1, &a_norm);
    t1m::bli::normv<T>(n, data_b, 1, &b_norm);

    const T estimator = norm_diff / (eps * d * a_norm * b_norm);

    EXPECT_NEAR(estimator, 0.f, 1e-6);

    alloc.deallocate(data_a, nelems_a);
    alloc.deallocate(data_b, nelems_b);
    alloc.deallocate(data_c, nelems_c);
  }
}

}  // namespace

///-----------------------------------------------------------------------------
///                      \n Real \n
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