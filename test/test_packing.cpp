#include <gtest/gtest.h>
#include "t1m/internal/packing.h"
#include "t1m/internal/scatter.h"
#include "t1m/internal/tensor.h"

using namespace t1m;

TEST(PackingTest, PackColMajorQuadratic) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;
  constexpr std::size_t sz = M * K;

  constexpr std::size_t m = 2;
  constexpr std::size_t k = 2;
  constexpr std::size_t block_size = m * k;

  std::array<float, sz> elems{};
  for (std::size_t i = 0; i < sz; ++i) {
    elems[i] = i + 1;
  }

  // 3D Layout:
  // 1 3 | 5 7 |  9 11 | 13 15
  // 2 4 | 6 8 | 10 12 | 14 16
  tensor<float, 3> t{{2, 2, 4}, elems.data(), memory_layout::COL_MAJOR};

  // 2D Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16
  block_layout layout(t.dimensions, t.strides(), {0, 1}, {2}, m, k);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Column Major:
  // 1  5   9  13
  // 2  6  10  14
  // ------------
  // 3  7  11  15
  // 4  8  12  16
  std::array<float, sz> pckd{};
  pack_block_col_major<float>(block, elems.data(), pckd.data());

  std::array<float, sz> expt{1, 2, 5, 6, 9,  10, 13, 14,
                             3, 4, 7, 8, 11, 12, 15, 16};
  EXPECT_TRUE(std::equal(pckd.data(), pckd.data() + sz, expt.begin()));
}

TEST(PackingTest, PackColMajorRectangular) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;
  constexpr std::size_t sz = M * K;

  constexpr std::size_t m = 3;
  constexpr std::size_t k = 2;
  constexpr std::size_t block_size = m * k;
  constexpr std::size_t space = 4 * block_size;

  std::array<float, sz> elems{};
  for (std::size_t i = 0; i < sz; ++i) {
    elems[i] = i + 1;
  }

  // 3D Layout:
  // 1 3 | 5 7 |  9 11 | 13 15
  // 2 4 | 6 8 | 10 12 | 14 16
  t1m::tensor<float, 3> t{{2, 2, 4}, elems.data(), memory_layout::COL_MAJOR};

  // 2D Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16

  block_layout layout(t.dimensions, t.strides(), {0, 1}, {2}, m, k);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Column Major:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // ------------
  // 4  8  12  16
  // 0  0   0   0
  // 0  0   0   0

  std::array<float, space> pckd{};
  pack_block_col_major<float>(block, elems.data(), pckd.data());

  std::array<float, space> expt{1, 2, 3, 5, 6, 7, 9,  10, 11, 13, 14, 15,
                                4, 0, 0, 8, 0, 0, 12, 0,  0,  16, 0,  0};
  EXPECT_TRUE(std::equal(pckd.data(), pckd.data() + sz, expt.begin()));
}

TEST(PackingTest, PackRowMajorQuadratic) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;
  constexpr std::size_t sz = M * K;

  constexpr std::size_t m = 2;
  constexpr std::size_t k = 2;
  constexpr std::size_t block_size = m * k;
  constexpr std::size_t space = sz;

  std::array<float, sz> elems{};
  for (std::size_t i = 0; i < sz; ++i) {
    elems[i] = i + 1;
  }

  // 3D Layout:
  // 1 3 | 5 7 |  9 11 | 13 15
  // 2 4 | 6 8 | 10 12 | 14 16
  tensor<float, 3> t{{2, 2, 4}, elems.data(), memory_layout::COL_MAJOR};

  // 2D Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16
  block_layout layout(t.dimensions, t.strides(), {0, 1}, {2}, m, k);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Row Major:
  // 1  5 |  9  13
  // 2  6 | 10  14
  // 3  7 | 11  15
  // 4  8 | 12  16
  std::array<float, space> pckd{};
  pack_block_row_major(block, elems.data(), pckd.data());

  std::array<float, space> expt{1, 5,  2,  6,  3,  7,  4,  8,
                                9, 13, 10, 14, 11, 15, 12, 16};
  EXPECT_TRUE(std::equal(pckd.data(), pckd.data() + sz, expt.begin()));
}

TEST(PackingTest, PackRowMajorRectangular) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;
  constexpr std::size_t sz = M * K;

  constexpr std::size_t m = 2;
  constexpr std::size_t k = 3;
  constexpr std::size_t block_size = m * k;
  constexpr std::size_t space = 4 * block_size;

  std::array<float, sz> elems{};
  for (std::size_t i = 0; i < sz; ++i) {
    elems[i] = i + 1;
  }

  // 3D Layout:
  // 1 3 | 5 7 |  9 11 | 13 15
  // 2 4 | 6 8 | 10 12 | 14 16
  t1m::tensor<float, 3> t{{2, 2, 4}, elems.data(), memory_layout::COL_MAJOR};

  // 2D Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16

  block_layout layout(t.dimensions, t.strides(), {0, 1}, {2}, m, k);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Row Major:
  // 1  5   9  | 13  0  0
  // 2  6  10  | 14  0  0
  // 3  7  11  | 15  0  0
  // 4  8  12  | 16  0  0
  std::array<float, space> pckd{};
  pack_block_row_major<float>(block, elems.data(), pckd.data());

  std::array<float, space> expt{1,  5, 9, 2,  6, 10, 3,  7, 11, 4,  8, 12,
                                13, 0, 0, 14, 0, 0,  15, 0, 0,  16, 0, 0};
  EXPECT_TRUE(std::equal(pckd.data(), pckd.data() + sz, expt.begin()));
}