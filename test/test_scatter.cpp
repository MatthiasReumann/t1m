#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "t1m/scatter.hpp"
#include <algorithm>

TEST_CASE("Calculate scatter vector")
{
  std::vector<size_t> lengths = {3, 2};
  std::vector<size_t> strides = {12, 6};
  std::vector<size_t> scat = t1m::internal::calc_scatter(lengths, strides);
  std::vector<size_t> expected{0, 6, 12, 18, 24, 30};
  REQUIRE(std::equal(scat.begin(), scat.end(), expected.begin()));
}

TEST_CASE("block scatter 1")
{
  std::vector<size_t> scat = { 1,2,3,5,7,8 };
  const size_t b = 3;
  const auto bs = t1m::internal::calc_block_scatter(scat, b);

  REQUIRE(bs.at(0) == 1);
  REQUIRE(bs.at(1) == 0);
}

TEST_CASE("block scatter 2")
{
  std::vector<size_t> scat = { 1,2,3,5,7,8,2 };
  const size_t b = 3;
  const auto bs = t1m::internal::calc_block_scatter(scat, b);

  REQUIRE(bs.at(0) == 1);
  REQUIRE(bs.at(1) == 0);
  REQUIRE(bs.at(2) == 2);
}

TEST_CASE("block scatter 3")
{
  std::vector<size_t> scat = { 1,2,3,5,7,8,1,4 };
  const size_t b = 3;
  const auto bs = t1m::internal::calc_block_scatter(scat, b);

  REQUIRE(bs.at(0) == 1);
  REQUIRE(bs.at(1) == 0);
  REQUIRE(bs.at(2) == 3);
}