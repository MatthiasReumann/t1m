#include <gtest/gtest.h>
#include "t1m/internal/tensor.h"
#include "t1m/internal/utils.h"

using namespace t1m;

TEST(UtilsTest, IndexBundling1) {
  // Test: abd = abc . cd
  contraction_labels labels{"abc", "cd", "abd"};
  contraction indices(labels);

  EXPECT_EQ(indices.CI, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(indices.CJ, (std::vector<std::size_t>{2}));
  EXPECT_EQ(indices.AI, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(indices.AP, (std::vector<std::size_t>{2}));
  EXPECT_EQ(indices.BP, (std::vector<std::size_t>{0}));
  EXPECT_EQ(indices.BJ, (std::vector<std::size_t>{1}));
}

TEST(UtilsTest, IndexBundling2) {
  // Test: fa = abcde . bdcf
  contraction_labels labels{"abcde", "bdcf", "fae"};
  contraction indices(labels);

  EXPECT_EQ(indices.CI, (std::vector<std::size_t>{1, 2}));
  EXPECT_EQ(indices.CJ, (std::vector<std::size_t>{0}));
  EXPECT_EQ(indices.AI, (std::vector<std::size_t>{0, 4}));
  EXPECT_EQ(indices.AP, (std::vector<std::size_t>{1, 2, 3}));
  EXPECT_EQ(indices.BP, (std::vector<std::size_t>{0, 2, 1}));
  EXPECT_EQ(indices.BJ, (std::vector<std::size_t>{3}));
}

TEST(UtilsTest, ScatterVectors) {
  t1m::tensor<float, 4> t{{3, 2, 2, 3}, nullptr, memory_layout::COL_MAJOR};
  std::vector<std::size_t> rscat =
      scatter<4>{}({0, 1}, t.dimensions, t.strides());
  std::vector<std::size_t> cscat =
      scatter<4>{}({2, 3}, t.dimensions, t.strides());

  EXPECT_EQ(rscat, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(cscat, (std::vector<std::size_t>{0, 6, 12, 18, 24, 30}));
}

TEST(UtilsTest, BlockScatterVectors1) {
  t1m::tensor<float, 4> t{{3, 2, 2, 3}, nullptr, memory_layout::COL_MAJOR};
  std::vector<std::size_t> rscat =
      scatter<4>{}({0, 1}, t.dimensions, t.strides());
  std::vector<std::size_t> cscat =
      scatter<4>{}({2, 3}, t.dimensions, t.strides());
  std::vector<std::size_t> block_rscat = block_scatter{3}(rscat);
  std::vector<std::size_t> block_cscat = block_scatter{3}(cscat);

  EXPECT_EQ(block_rscat, (std::vector<std::size_t>{1, 1}));
  EXPECT_EQ(block_cscat, (std::vector<std::size_t>{6, 6}));
}

TEST(UtilsTest, BlockScatterVectors2) {
  std::vector<std::size_t> block_scat = block_scatter{3}({1, 2, 3, 5, 7, 8});
  EXPECT_EQ(block_scat, (std::vector<std::size_t>{1, 0}));
}

TEST(UtilsTest, BlockScatterVectors3) {
  std::vector<std::size_t> block_scat = block_scatter{3}({1, 2, 3, 5, 7, 8, 2});
  EXPECT_EQ(block_scat, (std::vector<std::size_t>{1, 0, 2}));
}

TEST(UtilsTest, BlockScatterVectors4) {
  std::vector<std::size_t> block_scat =
      block_scatter{3}({1, 2, 3, 5, 7, 8, 1, 4});
  EXPECT_EQ(block_scat, (std::vector<std::size_t>{1, 0, 3}));
}