#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include "doctest/doctest.h"
#include "t1m/internal/utils.h"
#include "t1m/scatter.hpp"

TEST_CASE("Index Bundling") {
  SUBCASE("bundle lengths") {
    constexpr t1m::utils::contraction_labels labels{"abd", "abc", "cd"};
    constexpr auto lengths = t1m::utils::get_bundle_lengths(labels);
    REQUIRE(lengths.P == 1);
    REQUIRE(lengths.I == 2);
    REQUIRE(lengths.J == 1);
  }

  SUBCASE("contraction indices (abd = abc . cd)") {
    constexpr t1m::utils::contraction_labels labels{"abd", "abc", "cd"};
    constexpr auto lengths = t1m::utils::get_bundle_lengths(labels);
    constexpr auto indices =
        t1m::utils::contraction_indices<lengths>(labels);
    REQUIRE(indices.CI == std::array<std::size_t, lengths.I>{0, 1});
    REQUIRE(indices.CJ == std::array<std::size_t, lengths.J>{2});
    REQUIRE(indices.AI == std::array<std::size_t, lengths.I>{0, 1});
    REQUIRE(indices.AP == std::array<std::size_t, lengths.P>{2});
    REQUIRE(indices.BP == std::array<std::size_t, lengths.P>{0});
    REQUIRE(indices.BJ == std::array<std::size_t, lengths.J>{1});
  }

  SUBCASE("contraction indices (fa = abcde . bdcf)") {
    constexpr t1m::utils::contraction_labels labels{"fae", "abcde", "bdcf"};
    constexpr auto lengths = t1m::utils::get_bundle_lengths(labels);
    constexpr auto indices =
        t1m::utils::contraction_indices<lengths>(labels);
    REQUIRE(indices.CI == std::array<std::size_t, lengths.I>{1, 2});
    REQUIRE(indices.CJ == std::array<std::size_t, lengths.J>{0});
    REQUIRE(indices.AI == std::array<std::size_t, lengths.I>{0, 4});
    REQUIRE(indices.AP == std::array<std::size_t, lengths.P>{1, 2, 3});
    REQUIRE(indices.BP == std::array<std::size_t, lengths.P>{0, 2, 1});
    REQUIRE(indices.BJ == std::array<std::size_t, lengths.J>{3});
  }
}

TEST_CASE("Scatter And Block Scatter Vectors") {
  SUBCASE("scatter vector") {
    std::vector<size_t> lengths = {3, 2};
    std::vector<size_t> strides = {12, 6};
    std::vector<size_t> scat = t1m::utils::calc_scatter(lengths, strides);
    std::vector<size_t> expected{0, 6, 12, 18, 24, 30};
    REQUIRE(std::equal(scat.begin(), scat.end(), expected.begin()));
  }

  SUBCASE("block scatter variant 1") {
    std::vector<size_t> scat = {1, 2, 3, 5, 7, 8};
    const size_t b = 3;
    const auto bs = t1m::utils::calc_block_scatter(scat, b);

    REQUIRE(bs.at(0) == 1);
    REQUIRE(bs.at(1) == 0);
  }

  SUBCASE("block scatter variant 2") {
    std::vector<size_t> scat = {1, 2, 3, 5, 7, 8, 2};
    const size_t b = 3;
    const auto bs = t1m::utils::calc_block_scatter(scat, b);

    REQUIRE(bs.at(0) == 1);
    REQUIRE(bs.at(1) == 0);
    REQUIRE(bs.at(2) == 2);
  }

  SUBCASE("block scatter variant 3") {
    std::vector<size_t> scat = {1, 2, 3, 5, 7, 8, 1, 4};
    const size_t b = 3;
    const auto bs = t1m::utils::calc_block_scatter(scat, b);

    REQUIRE(bs.at(0) == 1);
    REQUIRE(bs.at(1) == 0);
    REQUIRE(bs.at(2) == 3);
  }
}