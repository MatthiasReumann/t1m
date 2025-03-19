#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <array>
#include "doctest/doctest.h"
#include "t1m/internal/scatter.h"
#include "t1m/internal/utils.h"
#include "t1m/t1m.h"
#include "t1m/tensor.h"

using namespace t1m;

TEST_CASE("packing of A tensor") {
  SUBCASE("quadratic block") {
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 4;
    constexpr std::size_t workspace_size = M * K;
    packing::lhs<float> workspace(workspace_size);

    constexpr std::size_t m = 2;
    constexpr std::size_t k = 2;
    constexpr std::size_t block_size = m * k;

    std::array<float, workspace_size> elems{};
    for (std::size_t i = 0; i < workspace_size; ++i) {
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
    scatter::block_layout layout(t.dimensions, t.strides(), {0, 1}, {2}, m, k);
    scatter::matrix_view block{layout.rs,  layout.cs, layout.br,
                               layout.rbs, layout.bc, layout.cbs};

    // Expected Packed Layout In Column Major:
    // 1  5   9  13
    // 2  6  10  14
    // ------------
    // 3  7  11  15
    // 4  8  12  16
    std::array<float, workspace_size> expected{1, 2, 5, 6, 9,  10, 13, 14,
                                               3, 4, 7, 8, 11, 12, 15, 16};
    workspace.pack_block(block, elems.data(), M, K);

    for (std::size_t i = 0; i < workspace_size; i++) {
      std::print(" {} ", workspace.data()[i]);
    }
    std::println("");
    REQUIRE(std::equal(workspace.data(), workspace.data() + workspace_size,
                       expected.begin()));
  }

  SUBCASE("rectangular block") {
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 4;
    constexpr std::size_t workspace_size = M * K;
    packing::lhs<float> workspace(workspace_size);

    constexpr std::size_t m = 3;
    constexpr std::size_t k = 2;
    constexpr std::size_t block_size = m * k;

    std::array<float, workspace_size> elems{};
    for (std::size_t i = 0; i < workspace_size; ++i) {
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
    scatter::block_layout layout(t.dimensions, t.strides(), {0, 1}, {2}, m, k);
    scatter::matrix_view block{layout.rs,  layout.cs, layout.br,
                               layout.rbs, layout.bc, layout.cbs};

    // Expected Packed Layout In Column Major:
    // 1  5   9  13
    // 2  6  10  14
    // 3  7  11  15
    // ------------
    // 4  8  12  16
    std::array<float, workspace_size> expected{1,  2,  3,  5,  6, 7, 9,  10,
                                               11, 13, 14, 15, 4, 8, 12, 16};
    workspace.pack_block(block, elems.data(), M, K);

    for (std::size_t i = 0; i < workspace_size; i++) {
      std::print(" {} ", workspace.data()[i]);
    }
    std::println("");
    REQUIRE(std::equal(workspace.data(), workspace.data() + workspace_size,
                       expected.begin()));
  }
}