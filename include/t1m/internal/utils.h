#pragma once

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace t1m::utils {

struct contraction {
  const std::string label_c;
  const std::string label_a;
  const std::string label_b;

  using bundle_lengths = std::tuple<std::size_t, std::size_t, std::size_t>;

  /// @brief Calculate bundle lengths for `P`, `I`, and `J`.
  /// @return `{ P, I, J }`
  constexpr bundle_lengths get_bundle_lengths() const noexcept {
    const std::size_t dC = label_c.size();
    const std::size_t dA = label_a.size();
    const std::size_t dB = label_b.size();
    const std::size_t P = (dA + dB - dC) / 2;
    return {P, dA - P, dB - P};
  }
};

template <const std::size_t P, const std::size_t I, const std::size_t J>
struct contraction_indices {
  constexpr contraction_indices(const contraction& spec) {
    std::string label_c(spec.label_c);
    std::string label_a(spec.label_a);
    std::string label_b(spec.label_b);

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

    for (size_t i = 0; i < contracted.size(); ++i) {
      AP[i] = spec.label_a.find(contracted[i]);
      BP[i] = spec.label_b.find(contracted[i]);
    }

    for (size_t i = 0; i < free_a.size(); ++i) {
      AI[i] = spec.label_a.find(free_a[i]);
      CI[i] = spec.label_c.find(free_a[i]);
    }

    for (size_t i = 0; i < free_b.size(); ++i) {
      BJ[i] = spec.label_b.find(free_b[i]);
      CJ[i] = spec.label_c.find(free_b[i]);
    }
  }

  std::array<std::size_t, I> CI;
  std::array<std::size_t, J> CJ;
  std::array<std::size_t, I> AI;
  std::array<std::size_t, P> AP;
  std::array<std::size_t, P> BP;
  std::array<std::size_t, J> BJ;
};

template <const std::size_t ndim>
struct scatter {
  const std::vector<std::size_t> indices;

  const std::array<std::size_t, ndim>& dimensions;
  const std::array<std::size_t, ndim>& strides;

  std::vector<std::size_t> operator()() const {
    const std::size_t N = indices.size();

    // compute scatter vectors.
    // first, create 2D vector with each index, stride combination.
    std::vector<std::vector<std::size_t>> parts{};
    for (std::size_t i = 0; i < N; ++i) {
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

template <std::size_t b>
struct block_scatter {
  const std::vector<std::size_t>& scat;

  std::vector<std::size_t> operator()() const {
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
