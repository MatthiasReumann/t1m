#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace t1m::internal
{
  std::vector<size_t> calc_block_scatter(std::vector<size_t>& scat, size_t b) {
    const size_t L = scat.size();
    const size_t N = std::ceil(L / double(b)); // ⌈l/b⌉

    std::vector<size_t> bs;
    bs.reserve(N);

    size_t j, last_stride, curr_stride;
    for (size_t i = 0; i < N; i++) {
      last_stride = 0;
      j = i * b;
      if (j + 1 == L) { // Edge case: Only one scat element in block.
        last_stride = scat.at(j);
      }
      else { // Normal case: More than one scat element in block.
        for (j = j; j < std::min<size_t>((i + 1) * b, L) - 1; j++) {
          curr_stride = scat.at(j + 1) - scat.at(j);

          if (last_stride != 0 && last_stride != curr_stride) {
            last_stride = 0;
            break; // Rest of elements in block are irrelevant.
          }
          else {
            last_stride = curr_stride;
          }
        }
      }
      bs.push_back(last_stride);
    }

    return bs;
  }
};
