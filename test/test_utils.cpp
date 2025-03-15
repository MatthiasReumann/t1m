#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include "doctest/doctest.h"
#include "t1m/internal/utils.h"
#include "t1m/t1m.h"
#include "t1m/tensor.h"

using namespace t1m;

TEST_CASE("index bundling") {
  SUBCASE("contraction indices (abd = abc . cd)") {
    constexpr utils::contraction spec{"abd", "abc", "cd"};
    constexpr auto bundle_lengths = spec.get_bundle_lengths();
    constexpr std::size_t P = std::get<0>(bundle_lengths);
    constexpr std::size_t I = std::get<1>(bundle_lengths);
    constexpr std::size_t J = std::get<2>(bundle_lengths);
    constexpr utils::contraction_indices<P, I, J> indices(spec);

    REQUIRE(indices.CI == std::array<std::size_t, I>{0, 1});
    REQUIRE(indices.CJ == std::array<std::size_t, J>{2});
    REQUIRE(indices.AI == std::array<std::size_t, I>{0, 1});
    REQUIRE(indices.AP == std::array<std::size_t, P>{2});
    REQUIRE(indices.BP == std::array<std::size_t, P>{0});
    REQUIRE(indices.BJ == std::array<std::size_t, J>{1});
  }

  SUBCASE("contraction indices (fa = abcde . bdcf)") {
    constexpr utils::contraction spec{"fae", "abcde", "bdcf"};
    constexpr auto bundle_lengths = spec.get_bundle_lengths();
    constexpr std::size_t P = std::get<0>(bundle_lengths);
    constexpr std::size_t I = std::get<1>(bundle_lengths);
    constexpr std::size_t J = std::get<2>(bundle_lengths);
    constexpr utils::contraction_indices<P, I, J> indices(spec);

    REQUIRE(indices.CI == std::array<std::size_t, I>{1, 2});
    REQUIRE(indices.CJ == std::array<std::size_t, J>{0});
    REQUIRE(indices.AI == std::array<std::size_t, I>{0, 4});
    REQUIRE(indices.AP == std::array<std::size_t, P>{1, 2, 3});
    REQUIRE(indices.BP == std::array<std::size_t, P>{0, 2, 1});
    REQUIRE(indices.BJ == std::array<std::size_t, J>{3});
  }
}

TEST_CASE("scatter vectors") {
  t1m::tensor<float, 4> t{{3, 2, 2, 3}, nullptr, memory_layout::COL_MAJOR};
  std::vector<std::size_t> rscat =
      utils::scatter{{0, 1}, t.dimensions, t.strides()}();
  std::vector<std::size_t> cscat =
      utils::scatter{{2, 3}, t.dimensions, t.strides()}();

  REQUIRE(rscat == std::vector<std::size_t>{0, 1, 2, 3, 4, 5});
  REQUIRE(cscat == std::vector<std::size_t>{0, 6, 12, 18, 24, 30});
}

TEST_CASE("block scatter vectors") {
  SUBCASE("variant 1") {
    t1m::tensor<float, 4> t{{3, 2, 2, 3}, nullptr, memory_layout::COL_MAJOR};
    std::vector<std::size_t> rscat =
        utils::scatter{{0, 1}, t.dimensions, t.strides()}();
    std::vector<std::size_t> cscat =
        utils::scatter{{2, 3}, t.dimensions, t.strides()}();
    std::vector<std::size_t> block_rscat = utils::block_scatter<3>{rscat}();
    std::vector<std::size_t> block_cscat = utils::block_scatter<3>{cscat}();

    REQUIRE(block_rscat == std::vector<std::size_t>{1, 1});
    REQUIRE(block_cscat == std::vector<std::size_t>{6, 6});
  }

  SUBCASE("variant 2") {
    std::vector<std::size_t> block_scat =
        utils::block_scatter<3>{{1, 2, 3, 5, 7, 8}}();
    REQUIRE(block_scat == std::vector<std::size_t>{1, 0});
  }

  SUBCASE("variant 3") {
    std::vector<std::size_t> block_scat =
        utils::block_scatter<3>{{1, 2, 3, 5, 7, 8, 2}}();
    REQUIRE(block_scat == std::vector<std::size_t>{1, 0, 2});
  }

  SUBCASE("variant 4") {
    std::vector<std::size_t> block_scat =
        utils::block_scatter<3>{{1, 2, 3, 5, 7, 8, 1, 4}}();
    REQUIRE(block_scat == std::vector<std::size_t>{1, 0, 3});
  }
}

TEST_CASE("contract") {
  t1m::tensor<float, 3> a{{3, 2, 2}, nullptr, memory_layout::COL_MAJOR};
  t1m::tensor<float, 2> b{{2, 3}, nullptr, memory_layout::COL_MAJOR};
  t1m::tensor<float, 3> c{{3, 2, 3}, nullptr, memory_layout::COL_MAJOR};

  // t1m::contract(c, "abd", a, "abc", b, "cd");
}