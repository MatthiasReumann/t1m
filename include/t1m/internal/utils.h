#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace t1m::utils {

struct contraction_labels {
  const std::string a;
  const std::string b;
  const std::string c;
};

struct contraction {
  contraction(const contraction_labels& labels) {
    std::string label_a(labels.a);
    std::string label_b(labels.b);
    std::string label_c(labels.c);

    std::sort(label_a.begin(), label_a.end());
    std::sort(label_b.begin(), label_b.end());
    std::sort(label_c.begin(), label_c.end());

    std::string contracted, free_a, free_b;
    std::set_intersection(label_a.begin(), label_a.end(), label_b.begin(),
                          label_b.end(), std::back_inserter(contracted));
    std::set_difference(label_a.begin(), label_a.end(), label_b.begin(),
                        label_b.end(), std::back_inserter(free_a));
    std::set_difference(label_b.begin(), label_b.end(), label_a.begin(),
                        label_a.end(), std::back_inserter(free_b));

    AP.reserve(contracted.size());
    BP.reserve(contracted.size());
    for (size_t i = 0; i < contracted.size(); ++i) {
      AP.push_back(labels.a.find(contracted[i]));
      BP.push_back(labels.b.find(contracted[i]));
    }

    AI.reserve(free_a.size());
    CI.reserve(free_a.size());
    for (size_t i = 0; i < free_a.size(); ++i) {
      AI.push_back(labels.a.find(free_a[i]));
      CI.push_back(labels.c.find(free_a[i]));
    }

    BJ.reserve(free_b.size());
    CJ.reserve(free_b.size());
    for (size_t i = 0; i < free_b.size(); ++i) {
      BJ.push_back(labels.b.find(free_b[i]));
      CJ.push_back(labels.c.find(free_b[i]));
    }
  }

  std::vector<std::size_t> CI;
  std::vector<std::size_t> CJ;

  std::vector<std::size_t> AI;
  std::vector<std::size_t> AP;

  std::vector<std::size_t> BP;
  std::vector<std::size_t> BJ;
};

template <const std::size_t ndim>
struct scatter {
  std::vector<std::size_t> operator()(
      const std::vector<std::size_t>& indices,
      const std::array<std::size_t, ndim>& dimensions,
      const std::array<std::size_t, ndim>& strides) const {
    // compute scatter vectors.
    // first, create 2D vector with each index, stride combination.
    std::vector<std::vector<std::size_t>> parts{};
    for (std::size_t i = 0; i < indices.size(); ++i) {
      const std::size_t& idx = indices[i];

      const std::size_t& dim = dimensions[idx];
      const std::size_t& stride = strides[idx];

      std::vector<std::size_t> v(dim);
      for (std::size_t j = 0; j < dim; ++j) {
        v[j] = j * stride;
      }
      parts.push_back(v);
    }

    // required to be equivalent with the old implementation.
    std::reverse(parts.begin(), parts.end());

    // second, reduce 2D to 1D vector, computing each possible combination.
    auto elementwise_add = [](std::vector<std::size_t> av,
                              std::vector<std::size_t> bv) {
      std::vector<std::size_t> cv;
      for (const std::size_t& a : av) {
        for (const std::size_t& b : bv) {
          cv.push_back(a + b);
        }
      }
      return cv;
    };

    return std::reduce(parts.begin() + 1, parts.end(), parts[0],
                       elementwise_add);
  }
};

struct block_scatter {
  const std::size_t b;

  std::vector<std::size_t> operator()(
      const std::vector<std::size_t>& scat) const {
    const size_t nblocks = (scat.size() + b - 1) / b;  // ⌈l/b⌉

    std::vector<std::size_t> block_scat(nblocks);
    for (std::size_t i = 0; i < nblocks; ++i) {
      const std::size_t offset = i * b;
      const std::size_t nelem = std::min<std::size_t>(b, scat.size() - offset);
      const std::vector<std::size_t> strides =
          get_strides(std::vector<std::size_t>(scat.begin() + offset,
                                               scat.begin() + offset + nelem));

      std::size_t stride = strides[0];
      if (!std::all_of(strides.begin(), strides.end(),
                       [stride](std::size_t i) { return i == stride; })) {
        stride = 0;
      }

      block_scat[i] = stride;
    }

    return block_scat;
  }

 private:
  std::vector<std::size_t> get_strides(
      const std::vector<std::size_t>& v) const {
    if (v.size() == 1) {
      return v;
    }

    std::vector<std::size_t> strides(v.size() - 1);
    for (std::size_t i = 0; i < v.size() - 1; ++i) {
      strides[i] = v[i + 1] - v[i];
    }
    return strides;
  }
};
};  // namespace t1m::utils
