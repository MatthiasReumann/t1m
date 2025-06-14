#include <gtest/gtest.h>
#include <algorithm>
#include <complex>
#include <cstddef>
#include "t1m/internal/packing.h"
#include "t1m/internal/scatter.h"
#include "t1m/tensor.h"

using namespace t1m;
using namespace t1m::internal;

namespace {
template <class T, std::size_t ndim>
void fill_complex(std::array<std::complex<T>, ndim>& elems) {
  for (std::size_t i = 0; i < elems.size(); ++i) {
    elems[i] = {static_cast<T>(2 * i + 1), static_cast<T>(2 * i + 2)};
  }
}
}  // namespace

///-----------------------------------------------------------------------------
///                      \n real packing \n
///-----------------------------------------------------------------------------

TEST(PackingTest, PackColMajorEven) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;
  constexpr std::size_t m = 2;
  constexpr std::size_t k = 2;

  std::array<float, M * K> elems{};
  std::iota(elems.begin(), elems.end(), 1);

  // 3D Layout:
  // 1 3 | 5 7 |  9 11 | 13 15
  // 2 4 | 6 8 | 10 12 | 14 16
  tensor<float, 3> t{{2, 2, 4}, elems.data(), col_major};

  // 2D Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16
  matrix_layout layout(t, {{0, 1}, {2}}, m, k);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Column Major:
  // 1  5   9  13
  // 2  6  10  14
  // ------------
  // 3  7  11  15
  // 4  8  12  16
  std::array<float, M * K> dest{};
  pack_block<float, A>(block, K, t.data, dest.data());

  std::array<float, M * K> expt{1, 2, 5, 6, 9,  10, 13, 14,
                                3, 4, 7, 8, 11, 12, 15, 16};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

TEST(PackingTest, PackColMajorOdd) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;
  constexpr std::size_t m = 3;
  constexpr std::size_t k = 2;

  std::array<float, M * K> elems{};
  std::iota(elems.begin(), elems.end(), 1);

  // 3D Layout:
  // 1 3 | 5 7 |  9 11 | 13 15
  // 2 4 | 6 8 | 10 12 | 14 16
  t1m::tensor<float, 3> t{{2, 2, 4}, elems.data(), col_major};

  // 2D Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16

  matrix_layout layout(t, {{0, 1}, {2}}, m, k);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Column Major:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // ------------
  // 4  8  12  16
  // 0  0   0   0
  // 0  0   0   0

  std::array<float, 2 * m * K> dest{};
  pack_block<float, A>(block, K, t.data, dest.data());

  std::array<float, 2 * m * K> expt{1, 2, 3, 5, 6, 7, 9,  10, 11, 13, 14, 15,
                                    4, 0, 0, 8, 0, 0, 12, 0,  0,  16, 0,  0};

  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

TEST(PackingTest, PackRowMajorEven) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;
  constexpr std::size_t m = 2;
  constexpr std::size_t k = 2;

  std::array<float, M * K> elems{};
  std::iota(elems.begin(), elems.end(), 1);

  // 3D Layout:
  // 1 3 | 5 7 |  9 11 | 13 15
  // 2 4 | 6 8 | 10 12 | 14 16
  tensor<float, 3> t{{2, 2, 4}, elems.data(), col_major};

  // 2D Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16
  matrix_layout layout(t, {{0, 1}, {2}}, m, k);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Row Major:
  // 1  5 |  9  13
  // 2  6 | 10  14
  // 3  7 | 11  15
  // 4  8 | 12  16
  std::array<float, M * K> dest{};
  pack_block<float, B>(block, K, t.data, dest.data());

  std::array<float, M * K> expt{1, 5,  2,  6,  3,  7,  4,  8,
                                9, 13, 10, 14, 11, 15, 12, 16};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

TEST(PackingTest, PackRowMajorOdd) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;
  constexpr std::size_t m = 2;
  constexpr std::size_t k = 3;

  std::array<float, M * K> elems{};
  std::iota(elems.begin(), elems.end(), 1);

  // 3D Layout:
  // 1 3 | 5 7 |  9 11 | 13 15
  // 2 4 | 6 8 | 10 12 | 14 16
  t1m::tensor<float, 3> t{{2, 2, 4}, elems.data(), col_major};

  // 2D Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16

  matrix_layout layout(t, {{0, 1}, {2}}, m, k);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Row Major:
  // 1  5   9  | 13  0  0
  // 2  6  10  | 14  0  0
  // 3  7  11  | 15  0  0
  // 4  8  12  | 16  0  0
  std::array<float, M * 2 * k> dest{};
  pack_block<float, B>(block, K, t.data, dest.data());

  std::array<float, M * 2 * k> expt{1,  5, 9, 2,  6, 10, 3,  7, 11, 4,  8, 12,
                                    13, 0, 0, 14, 0, 0,  15, 0, 0,  16, 0, 0};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

TEST(PackingTest, Unpack) {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 4;

  std::array<float, M * K> elems{};
  std::iota(elems.begin(), elems.end(), 1);

  // Tensor Layout:
  // 1  5   9  13
  // 2  6  10  14
  // 3  7  11  15
  // 4  8  12  16
  tensor<float, 3> t{{M, K}, elems.data(), col_major};

  // 2D Layout:
  //  1   2   3   4
  //  5   6   7   8
  //  9  10  11  12
  // 13  14  15  16
  matrix_layout layout(t, {{1}, {0}}, M, K);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Unpacked Layout
  //  1   2   3   4
  //  5   6   7   8
  //  9  10  11  12
  // 13  14  15  16
  std::array<float, M * K> dest{};
  unpack(block, t.data, dest.data());

  std::array<float, M * K> expt{1, 5, 9,  13, 2, 6, 10, 14,
                                3, 7, 11, 15, 4, 8, 12, 16};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

///-----------------------------------------------------------------------------
///                      \n complex (1m) packing \n
///-----------------------------------------------------------------------------

TEST(PackingTest, Pack1MColMajorEven) {
  constexpr std::size_t X = 4;  // 4 * 4 complex numbers.
  constexpr std::size_t Y = 4;

  constexpr std::size_t MR = 4;  // 4 * 8 real slivers.
  constexpr std::size_t KP = 8;

  std::array<std::complex<float>, X * Y> elems{};
  fill_complex<float>(elems);

  // 3D Layout:
  // 1+2i 5+6i |  9+10i 13+14i |  17+18i 21+22i | 25+26i 29+30i
  // 3+4i 7+8i | 11+12i 15+16i |  19+20i 23+24i | 27+28i 31+32i
  tensor<std::complex<float>, 3> t{{2, 2, 4}, elems.data(), col_major};

  // 2D Layout
  // 1+2i   9+10i  17+18i  25+26i
  // 3+4i  11+12i  19+20i  27+28i
  // 5+6i  13+14i  21+22i  29+30i
  // 7+8i  15+16i  23+24i  31+32i
  matrix_layout layout(t, {{0, 1}, {2}}, MR, KP);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Column Major:
  // 1 -2   9 -10  17 -18  25 -26
  // 2  1  10   9  18  17  26  25
  // 3 -4  11 -12  19 -20  27 -28
  // 4  3  12  11  20  19  28  27
  // ---------------------------
  // 5 -6  13 -14  21 -22  29 -30
  // 6  5  14  13  22  21  30  29
  // 7 -8  15 -16  23 -24  31 -32
  // 8  7  16  15  24  23  32  31
  std::array<float, 2 * MR * KP> dest{};
  pack_block<std::complex<float>, A>(block, KP, t.data, dest.data());

  std::array<float, 2 * MR * KP> expt{
      1,  2,  3,  4,  -2,  1,  -4,  3,  9,  10, 11, 12, -10, 9,  -12, 11,
      17, 18, 19, 20, -18, 17, -20, 19, 25, 26, 27, 28, -26, 25, -28, 27,
      5,  6,  7,  8,  -6,  5,  -8,  7,  13, 14, 15, 16, -14, 13, -16, 15,
      21, 22, 23, 24, -22, 21, -24, 23, 29, 30, 31, 32, -30, 29, -32, 31};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

TEST(PackingTest, Pack1MColMajorOdd) {
  constexpr std::size_t X = 4;  // 4 * 4 complex numbers.
  constexpr std::size_t Y = 4;

  constexpr std::size_t MR = 6;  // 4 * 8 real slivers.
  constexpr std::size_t KP = 8;

  std::array<std::complex<float>, X * Y> elems{};
  fill_complex<float>(elems);

  // 3D Layout:
  // 1+2i 5+6i |  9+10i 13+14i |  17+18i 21+22i | 25+26i 29+30i
  // 3+4i 7+8i | 11+12i 15+16i |  19+20i 23+24i | 27+28i 31+32i
  tensor<std::complex<float>, 3> t{{2, 2, 4}, elems.data(), col_major};

  // 2D Layout
  // 1+2i   9+10i  17+18i  25+26i
  // 3+4i  11+12i  19+20i  27+28i
  // 5+6i  13+14i  21+22i  29+30i
  // 7+8i  15+16i  23+24i  31+32i
  matrix_layout layout(t, {{0, 1}, {2}}, MR, KP);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Column Major:
  // 1 -2   9 -10  17 -18  25 -26
  // 2  1  10   9  18  17  26  25
  // 3 -4  11 -12  19 -20  27 -28
  // 4  3  12  11  20  19  28  27
  // 5 -6  13 -14  21 -22  29 -30
  // 6  5  14  13  22  21  30  29
  // ---------------------------
  // 7 -8  15 -16  23 -24  31 -32
  // 8  7  16  15  24  23  32  31
  // 0  0   0   0   0   0   0   0
  // 0  0   0   0   0   0   0   0
  // 0  0   0   0   0   0   0   0
  // 0  0   0   0   0   0   0   0
  std::array<float, 2 * MR * KP> dest{};
  pack_block<std::complex<float>, A>(block, KP, t.data, dest.data());

  std::array<float, 2 * MR * KP> expt{
      1,   2,  3,   4,  5,   6,  -2,  1,  -4, 3,  -6,  5,  9,   10, 11,  12,
      13,  14, -10, 9,  -12, 11, -14, 13, 17, 18, 19,  20, 21,  22, -18, 17,
      -20, 19, -22, 21, 25,  26, 27,  28, 29, 30, -26, 25, -28, 27, -30, 29,
      7,   8,  0,   0,  0,   0,  -8,  7,  0,  0,  0,   0,  15,  16, 0,   0,
      0,   0,  -16, 15, 0,   0,  0,   0,  23, 24, 0,   0,  0,   0,  -24, 23,
      0,   0,  0,   0,  31,  32, 0,   0,  0,  0,  -32, 31, 0,   0,  0,   0};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

TEST(PackingTest, Pack1MRowMajorEven) {
  constexpr std::size_t X = 4;  // 4 * 4 complex numbers.
  constexpr std::size_t Y = 4;

  constexpr std::size_t KP = 8;  // 8 * 2 real slivers.
  constexpr std::size_t NR = 2;

  std::array<std::complex<float>, X * Y> elems{};
  fill_complex<float>(elems);

  // 3D Layout:
  // 1+2i 5+6i |  9+10i 13+14i |  17+18i 21+22i | 25+26i 29+30i
  // 3+4i 7+8i | 11+12i 15+16i |  19+20i 23+24i | 27+28i 31+32i
  tensor<std::complex<float>, 3> t{{2, 2, 4}, elems.data(), col_major};

  // 2D Layout
  // 1+2i   9+10i  17+18i  25+26i
  // 3+4i  11+12i  19+20i  27+28i
  // 5+6i  13+14i  21+22i  29+30i
  // 7+8i  15+16i  23+24i  31+32i
  matrix_layout layout(t, {{0, 1}, {2}}, KP, NR);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Row Major:
  // 1  9 | 17 25
  // 2 10 | 18 26
  // 3 11 | 19 27
  // 4 12 | 20 28
  // 5 13 | 21 29
  // 6 14 | 22 30
  // 7 15 | 23 31
  // 8 16 | 24 32
  std::array<float, 2 * KP * NR> dest{};
  pack_block<std::complex<float>, B>(block, KP, t.data, dest.data());

  std::array<float, 2 * KP * NR> expt{
      1,  9,  2,  10, 3,  11, 4,  12, 5,  13, 6,  14, 7,  15, 8,  16,
      17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 23, 31, 24, 32};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

TEST(PackingTest, Pack1MRowMajorOdd) {
  constexpr std::size_t X = 4;  // 4 * 4 complex numbers.
  constexpr std::size_t Y = 4;

  constexpr std::size_t KP = 8;  // 8 * 3 real slivers.
  constexpr std::size_t NR = 3;

  std::array<std::complex<float>, X * Y> elems{};
  fill_complex<float>(elems);

  // 3D Layout:
  // 1+2i 5+6i |  9+10i 13+14i |  17+18i 21+22i | 25+26i 29+30i
  // 3+4i 7+8i | 11+12i 15+16i |  19+20i 23+24i | 27+28i 31+32i
  tensor<std::complex<float>, 3> t{{2, 2, 4}, elems.data(), col_major};

  // 2D Layout
  // 1+2i   9+10i  17+18i  25+26i
  // 3+4i  11+12i  19+20i  27+28i
  // 5+6i  13+14i  21+22i  29+30i
  // 7+8i  15+16i  23+24i  31+32i
  matrix_layout layout(t, {{0, 1}, {2}}, KP, NR);
  matrix_view block = matrix_view::from_layout(layout);

  // Expected Packed Layout In Row Major:
  // 1  9 17 | 25 0 0
  // 2 10 18 | 26 0 0
  // 3 11 19 | 27 0 0
  // 4 12 20 | 28 0 0
  // 5 13 21 | 29 0 0
  // 6 14 22 | 30 0 0
  // 7 15 23 | 31 0 0
  // 8 16 24 | 32 0 0
  std::array<float, 2 * KP * NR> dest{};
  pack_block<std::complex<float>, B>(block, KP, t.data, dest.data());

  std::array<float, 2 * KP * NR> expt{
      1,  9,  17, 2,  10, 18, 3,  11, 19, 4, 12, 20, 5, 13, 21, 6,
      14, 22, 7,  15, 23, 8,  16, 24, 25, 0, 0,  26, 0, 0,  27, 0,
      0,  28, 0,  0,  29, 0,  0,  30, 0,  0, 31, 0,  0, 32, 0,  0};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}

TEST(PackingTest, Unpack1M) {
  constexpr std::size_t X = 4;
  constexpr std::size_t Y = 4;

  std::array<std::complex<float>, X * Y> elems{};
  fill_complex<float>(elems);

  // Tensor Layout:
  // 1+2i   9+10i  17+18i  25+26i
  // 3+4i  11+12i  19+20i  27+28i
  // 5+6i  13+14i  21+22i  29+30i
  // 7+8i  15+16i  23+24i  31+32i
  tensor<std::complex<float>, 3> t{{X, Y}, elems.data(), col_major};

  // 2D Layout
  //  1+ 2i   3+ 4i   5+ 6i   7+ 8i
  //  9+10i  11+12i  13+14i  15+16i
  // 17+18i  19+20i  21+22i  23+24i
  // 25+26i  27+28i  29+30i  31+32i
  matrix_layout layout(t, {{1}, {0}}, 2 * X, 2 * Y);
  matrix_view block = matrix_view::from_layout(layout);

  std::array<std::complex<float>, X * Y> dest{};
  unpack(block, reinterpret_cast<float*>(t.data), dest.data());

  std::array<std::complex<float>, X * Y> expt{std::complex<float>{1, 2},
                                              {9, 10},
                                              {17, 18},
                                              {25, 26},
                                              {3, 4},
                                              {11, 12},
                                              {19, 20},
                                              {27, 28},
                                              {5, 6},
                                              {13, 14},
                                              {21, 22},
                                              {29, 30},
                                              {7, 8},
                                              {15, 16},
                                              {23, 24},
                                              {31, 32}};
  EXPECT_TRUE(std::equal(dest.begin(), dest.end(), expt.begin()));
}