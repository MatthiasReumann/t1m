#include <gtest/gtest.h>
#include "t1m/internal/scatter.h"
#include "t1m/tensor.h"

using namespace t1m;
using namespace t1m::internal;

TEST(UtilsTest, IndexBundling1) {
  // Test: abd = abc . cd
  const auto [A, B, C] = get_index_bundles("abc", "cd", "abd");

  EXPECT_EQ(C.X, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(C.Y, (std::vector<std::size_t>{2}));
  EXPECT_EQ(A.X, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(A.Y, (std::vector<std::size_t>{2}));
  EXPECT_EQ(B.X, (std::vector<std::size_t>{0}));
  EXPECT_EQ(B.Y, (std::vector<std::size_t>{1}));
}

TEST(UtilsTest, IndexBundling2) {
  // Test: fa = abcde . bdcf
  const auto [A, B, C] = get_index_bundles("abcde", "bdcf", "fae");

  EXPECT_EQ(C.X, (std::vector<std::size_t>{1, 2}));
  EXPECT_EQ(C.Y, (std::vector<std::size_t>{0}));
  EXPECT_EQ(A.X, (std::vector<std::size_t>{0, 4}));
  EXPECT_EQ(A.Y, (std::vector<std::size_t>{1, 2, 3}));
  EXPECT_EQ(B.X, (std::vector<std::size_t>{0, 2, 1}));
  EXPECT_EQ(B.Y, (std::vector<std::size_t>{3}));
}

TEST(UtilsTest, ScatterVectors) {
  t1m::tensor<float, 4> t{{3, 2, 2, 3}, nullptr, memory_layout::col_major};
  std::vector<std::size_t> rscat = get_scatter<4>({0, 1}, t.dims, t.strides());
  std::vector<std::size_t> cscat = get_scatter<4>({2, 3}, t.dims, t.strides());

  EXPECT_EQ(rscat, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(cscat, (std::vector<std::size_t>{0, 6, 12, 18, 24, 30}));
}

TEST(UtilsTest, BlockScatterVectors1) {
  t1m::tensor<float, 4> t{{3, 2, 2, 3}, nullptr, memory_layout::col_major};
  std::vector<std::size_t> rscat = get_scatter<4>({0, 1}, t.dims, t.strides());
  std::vector<std::size_t> cscat = get_scatter<4>({2, 3}, t.dims, t.strides());
  std::vector<std::size_t> block_rscat = get_block_scatter(rscat, 3);
  std::vector<std::size_t> block_cscat = get_block_scatter(cscat, 3);

  EXPECT_EQ(block_rscat, (std::vector<std::size_t>{1, 1}));
  EXPECT_EQ(block_cscat, (std::vector<std::size_t>{6, 6}));
}

TEST(UtilsTest, BlockScatterVectors2) {
  std::vector<std::size_t> block_scat =
      get_block_scatter({1, 2, 3, 5, 7, 8}, 3);
  EXPECT_EQ(block_scat, (std::vector<std::size_t>{1, 0}));
}

TEST(UtilsTest, BlockScatterVectors3) {
  std::vector<std::size_t> block_scat =
      get_block_scatter({1, 2, 3, 5, 7, 8, 2}, 3);
  EXPECT_EQ(block_scat, (std::vector<std::size_t>{1, 0, 0}));
}

TEST(UtilsTest, BlockScatterVectors4) {
  std::vector<std::size_t> block_scat =
      get_block_scatter({1, 2, 3, 5, 7, 8, 1, 4}, 3);
  EXPECT_EQ(block_scat, (std::vector<std::size_t>{1, 0, 3}));
}