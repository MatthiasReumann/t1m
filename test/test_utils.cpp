#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include "doctest/doctest.h"
#include "t1m/internal/utils.h"
#include "t1m/scatter.hpp"

using namespace t1m;

TEST_CASE("index bundling") {
  SUBCASE("bundle lengths") {
    constexpr utils::contraction_labels labels{"abd", "abc", "cd"};
    constexpr auto lengths = ::utils::get_bundle_lengths(labels);

    REQUIRE(lengths.P == 1);
    REQUIRE(lengths.I == 2);
    REQUIRE(lengths.J == 1);
  }

  SUBCASE("contraction indices (abd = abc . cd)") {
    constexpr utils::contraction_labels labels{"abd", "abc", "cd"};
    constexpr auto lengths = utils::get_bundle_lengths(labels);
    constexpr auto indices = utils::contraction_indices<lengths>(labels);

    REQUIRE(indices.CI == std::array<std::size_t, lengths.I>{0, 1});
    REQUIRE(indices.CJ == std::array<std::size_t, lengths.J>{2});
    REQUIRE(indices.AI == std::array<std::size_t, lengths.I>{0, 1});
    REQUIRE(indices.AP == std::array<std::size_t, lengths.P>{2});
    REQUIRE(indices.BP == std::array<std::size_t, lengths.P>{0});
    REQUIRE(indices.BJ == std::array<std::size_t, lengths.J>{1});
  }

  SUBCASE("contraction indices (fa = abcde . bdcf)") {
    constexpr utils::contraction_labels labels{"fae", "abcde", "bdcf"};
    constexpr auto lengths = utils::get_bundle_lengths(labels);
    constexpr auto indices = utils::contraction_indices<lengths>(labels);

    REQUIRE(indices.CI == std::array<std::size_t, lengths.I>{1, 2});
    REQUIRE(indices.CJ == std::array<std::size_t, lengths.J>{0});
    REQUIRE(indices.AI == std::array<std::size_t, lengths.I>{0, 4});
    REQUIRE(indices.AP == std::array<std::size_t, lengths.P>{1, 2, 3});
    REQUIRE(indices.BP == std::array<std::size_t, lengths.P>{0, 2, 1});
    REQUIRE(indices.BJ == std::array<std::size_t, lengths.J>{3});
  }
}

TEST_CASE("scatter vectors") {
  constexpr std::size_t ndim = 4;
  constexpr std::array<std::size_t, ndim> dimensions{3, 2, 2, 3};
  constexpr std::array<std::size_t, ndim> strides =
      utils::compute_strides(dimensions, utils::memory_layout::COL_MAJOR);
  std::vector<std::size_t> rscat =
      utils::scatter{{0, 1}, dimensions, strides}();
  std::vector<std::size_t> cscat =
      utils::scatter{{2, 3}, dimensions, strides}();

  REQUIRE(rscat == std::vector<std::size_t>{0, 1, 2, 3, 4, 5});
  REQUIRE(cscat == std::vector<std::size_t>{0, 6, 12, 18, 24, 30});
}

TEST_CASE("block scatter vectors") {
  SUBCASE("variant 1") {
    constexpr std::size_t ndim = 4;
    constexpr std::array<std::size_t, ndim> dimensions{3, 2, 2, 3};
    constexpr std::array<std::size_t, ndim> strides =
        utils::compute_strides(dimensions, utils::memory_layout::COL_MAJOR);
    std::vector<std::size_t> rscat =
        utils::scatter{{0, 1}, dimensions, strides}();
    std::vector<std::size_t> cscat =
        utils::scatter{{2, 3}, dimensions, strides}();
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