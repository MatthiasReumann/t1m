#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <array>
#include "t1m/tensor.h"
#include "t1m/t1m.h"

TEST_CASE("ab . bc = ac") {
  std::array<float, 6> ad;
  std::array<float, 6> bd;
  std::array<float, 4> cd;

  t1m::tensor<float, 2> a{{2, 3}, ad.data(), t1m::memory_layout::COL_MAJOR};
  t1m::tensor<float, 2> b{{3, 2}, bd.data(), t1m::memory_layout::COL_MAJOR};
  t1m::tensor<float, 2> c{{2, 2}, cd.data(), t1m::memory_layout::COL_MAJOR};
  t1m::contract(1.f, a, "ab", b, "bc", 0.f, c, "ac");
}