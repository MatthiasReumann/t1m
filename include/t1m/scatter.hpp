#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "utils.hpp"
#include "marray.hpp"

namespace t1m::internal
{
  auto fuse_vectors(const std::vector<size_t> &v1, const std::vector<size_t> &v2) {
    std::vector<std::vector<size_t>> fused;
    fused.reserve(v1.size());

    for (const auto& s : v1)
    {
      std::vector<size_t> c(v2);
      std::for_each(c.begin(), c.end(), [&s](size_t& n) { n += s; });
      fused.push_back(c);
    }

    return fused;
  }

  auto calc_scatter_rec(std::vector<std::vector<size_t>> ls) {
    if (ls.size() == 1) // Until only one vector is left.
      return ls;

    std::vector<std::vector<size_t>> res;
    for (const auto& v : fuse_vectors(ls.at(0), ls.at(1)))
    {
      // Update working vector: Remove the two visited vectors.
      std::vector<std::vector<size_t>> work_vec { {v} };
      work_vec.insert(work_vec.end(), ls.begin() + 2, ls.end());

      // Recursive function call with working vector.
      auto scat_2D = calc_scatter_rec(work_vec);
      res.insert(res.end(), scat_2D.begin(), scat_2D.end());
    }

    return res;
  }

  auto calc_scatter(std::vector<size_t>& lengths, std::vector<size_t>& strides) {
    std::vector<size_t> scat;
    std::vector<std::vector<size_t>> ls;

    // Multiply lengths with strides.
    // Collect in 2D vector.
    for (size_t i = 0; i < lengths.size(); i++)
    {
      ls.emplace_back();
      for (size_t j = 0; j < lengths.at(i); j++)
        ls.at(i).push_back(j * strides.at(i));
    }

    // Flatten 2D vector to 1D vector
    for (const auto& v : calc_scatter_rec(ls))
      scat.insert(scat.end(), v.begin(), v.end());

    return scat;
  }

  auto calc_block_scatter(std::vector<size_t>& scat, size_t b) {
    const size_t L = scat.size();
    const size_t N = (L + b - 1) / b; // ⌈l/b⌉

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
        for (; j < std::min<size_t>((i + 1) * b, L) - 1; j++) {
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