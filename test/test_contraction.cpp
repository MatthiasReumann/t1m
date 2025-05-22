#include <blis/blis.h>
#include <gtest/gtest.h>
#include <functional>
#include <limits>
#include <numeric>
#include "t1m/internal/t1m.h"
#include "t1m/internal/tensor.h"

using namespace t1m;

TEST(ContractionTest, FloatMinimal) {
  constexpr memory_layout layout = col_major;

  // ┌ 1 3 5 ┐   ┌ 1 4 ┐   ┌ 22 49 ┐
  // └ 2 4 6 ┘ . | 2 5 | = └ 28 64 ┘
  //             └ 3 6 ┘
  std::array<float, 6> ad{1, 2, 3, 4, 5, 6};
  std::array<float, 6> bd{1, 2, 3, 4, 5, 6};
  std::array<float, 4> cd{0, 0, 0, 0};

  t1m::tensor<float, 2> a{{2, 3}, ad.data(), layout};
  t1m::tensor<float, 2> b{{3, 2}, bd.data(), layout};
  t1m::tensor<float, 2> c{{2, 2}, cd.data(), layout};

  t1m::contract(1.f, a, "ab", b, "bc", 0.f, c, "ac");

  EXPECT_EQ(cd, (std::array<float, 4>{22, 28, 49, 64}));
}

TEST(ContractionTest, DoubleMinimal) {
  constexpr memory_layout layout = col_major;

  // ┌ 1 3 5 ┐   ┌ 1 4 ┐   ┌ 22 49 ┐
  // └ 2 4 6 ┘ . | 2 5 | = └ 28 64 ┘
  //             └ 3 6 ┘
  std::array<double, 6> ad{1, 2, 3, 4, 5, 6};
  std::array<double, 6> bd{1, 2, 3, 4, 5, 6};
  std::array<double, 4> cd{0, 0, 0, 0};

  t1m::tensor<double, 2> a{{2, 3}, ad.data(), layout};
  t1m::tensor<double, 2> b{{3, 2}, bd.data(), layout};
  t1m::tensor<double, 2> c{{2, 2}, cd.data(), layout};

  t1m::contract(1., a, "ab", b, "bc", 0., c, "ac");

  EXPECT_EQ(cd, (std::array<double, 4>{22, 28, 49, 64}));
}

TEST(ContractionTest, FloatRandom) {
  constexpr std::size_t MAX_DIM = 20;
  constexpr memory_layout layout = col_major;
  constexpr float eps = std::numeric_limits<float>::epsilon();

  const float one = 1.f;
  const float zero = 0.f;
  const float mone = -1.f;

  std::allocator<float> alloc{};

  const auto multiply = [](const auto& dims) {
    return std::accumulate(dims.begin(), dims.end(), 1.f,
                           std::multiplies<float>());
  };

  for (std::size_t d = 2; d <= MAX_DIM; ++d) {
    const std::array<std::size_t, 3> a_dims{d, d, d};
    const std::array<std::size_t, 2> b_dims{d, d};
    const std::array<std::size_t, 3> c_dims{d, d, d};

    const std::size_t a_nelems = multiply(a_dims);
    const std::size_t b_nelems = multiply(b_dims);
    const std::size_t c_nelems = multiply(c_dims);

    float* a_ptr = alloc.allocate(a_nelems);
    float* b_ptr = alloc.allocate(b_nelems);
    float* c_ptr = alloc.allocate(c_nelems);
    float* c_ref_ptr = alloc.allocate(c_nelems);

    bli_srandv(a_nelems, a_ptr, 1);
    bli_srandv(b_nelems, b_ptr, 1);
    bli_ssetv(BLIS_NO_CONJUGATE, c_nelems, &zero, c_ptr, 1);
    bli_ssetv(BLIS_NO_CONJUGATE, c_nelems, &zero, c_ref_ptr, 1);

    t1m::tensor<float, 3> a{a_dims, a_ptr, layout};
    t1m::tensor<float, 2> b{b_dims, b_ptr, layout};
    t1m::tensor<float, 3> c{c_dims, c_ptr, layout};

    t1m::contract(1.f, a, "abc", b, "cd", 0.f, c, "abd");

    const std::size_t m = d * d;
    const std::size_t n = d;
    const std::size_t k = d;

    // clang-format off
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
      m, n, k, 
      &one, 
      a_ptr, 1, m, 
      b_ptr, 1, k,
      &zero,
      c_ref_ptr, 1, m);
     
    bli_saxpym(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
              m, n, &mone,
              c_ptr, 1, m, 
              c_ref_ptr, 1, m);
    // clang-format on

    float norm_diff, a_norm, b_norm;
    bli_snormfv(n, c_ref_ptr, 1, &norm_diff);
    bli_snormfv(n, a_ptr, 1, &a_norm);
    bli_snormfv(n, b_ptr, 1, &b_norm);

    auto estimator = norm_diff / (eps * d * a_norm * b_norm);

    EXPECT_NEAR(estimator, 0.f, 1e-6);

    alloc.deallocate(a_ptr, a_nelems);
    alloc.deallocate(b_ptr, b_nelems);
    alloc.deallocate(c_ptr, c_nelems);
  }
}

TEST(ContractionTest, DoubleRandom) {
  constexpr std::size_t MAX_DIM = 20;
  constexpr memory_layout layout = col_major;
  constexpr double eps = std::numeric_limits<double>::epsilon();

  const double one = 1.f;
  const double zero = 0.f;
  const double mone = -1.f;

  std::allocator<double> alloc{};

  const auto multiply = [](const auto& dims) {
    return std::accumulate(dims.begin(), dims.end(), 1.f,
                           std::multiplies<double>());
  };

  for (std::size_t d = 2; d <= MAX_DIM; ++d) {
    const std::array<std::size_t, 3> a_dims{d, d, d};
    const std::array<std::size_t, 2> b_dims{d, d};
    const std::array<std::size_t, 3> c_dims{d, d, d};

    const std::size_t a_nelems = multiply(a_dims);
    const std::size_t b_nelems = multiply(b_dims);
    const std::size_t c_nelems = multiply(c_dims);

    double* a_ptr = alloc.allocate(a_nelems);
    double* b_ptr = alloc.allocate(b_nelems);
    double* c_ptr = alloc.allocate(c_nelems);
    double* c_ref_ptr = alloc.allocate(c_nelems);

    bli_drandv(a_nelems, a_ptr, 1);
    bli_drandv(b_nelems, b_ptr, 1);
    bli_dsetv(BLIS_NO_CONJUGATE, c_nelems, &zero, c_ptr, 1);
    bli_dsetv(BLIS_NO_CONJUGATE, c_nelems, &zero, c_ref_ptr, 1);

    t1m::tensor<double, 3> a{a_dims, a_ptr, layout};
    t1m::tensor<double, 2> b{b_dims, b_ptr, layout};
    t1m::tensor<double, 3> c{c_dims, c_ptr, layout};

    t1m::contract(1., a, "abc", b, "cd", 0., c, "abd");

    const std::size_t m = d * d;
    const std::size_t n = d;
    const std::size_t k = d;

    // clang-format off
    bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
      m, n, k,
      &one,
      a_ptr, 1, m,
      b_ptr, 1, k,
      &zero,
      c_ref_ptr, 1, m);

    bli_daxpym(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
              m, n, &mone,
              c_ptr, 1, m,
              c_ref_ptr, 1, m);
    // clang-format on

    double norm_diff, a_norm, b_norm;
    bli_dnormfv(n, c_ref_ptr, 1, &norm_diff);
    bli_dnormfv(n, a_ptr, 1, &a_norm);
    bli_dnormfv(n, b_ptr, 1, &b_norm);

    auto estimator = norm_diff / (eps * d * a_norm * b_norm);

    EXPECT_NEAR(estimator, 0., 1e-6);

    alloc.deallocate(a_ptr, a_nelems);
    alloc.deallocate(b_ptr, b_nelems);
    alloc.deallocate(c_ptr, c_nelems);
  }
}