#include <gtest/gtest.h>
#include "t1m/internal/t1m.h"
#include "t1m/internal/tensor.h"

using namespace t1m;

TEST(ContractionTest, Real) {
  std::array<float, 6> ad{1, 2, 3, 4, 5, 6};
  std::array<float, 6> bd{1, 2, 3, 4, 5, 6};
  std::array<float, 4> cd{0, 0, 0, 0};

  t1m::tensor<float, 2> a{{2, 3}, ad.data(), t1m::memory_layout::COL_MAJOR};
  t1m::tensor<float, 2> b{{3, 2}, bd.data(), t1m::memory_layout::COL_MAJOR};
  t1m::tensor<float, 2> c{{2, 2}, cd.data(), t1m::memory_layout::COL_MAJOR};
  t1m::contract(1.f, a, "ab", b, "bc", 0.f, c, "ac");
  EXPECT_FALSE(true);
}