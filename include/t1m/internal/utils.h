#pragma once

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <vector>
#include "../tensor.hpp"

namespace t1m::utils {

struct contraction_labels {
  const std::string c;
  const std::string a;
  const std::string b;
};

struct bundle_lengths {
  const std::size_t P;
  const std::size_t I;
  const std::size_t J;
};

consteval bundle_lengths get_bundle_lengths(const contraction_labels& labels) {
  const std::size_t dC = labels.c.size();
  const std::size_t dA = labels.a.size();
  const std::size_t dB = labels.b.size();

  const std::size_t length_p = (dA + dB - dC) / 2;

  return {length_p, dA - length_p, dB - length_p};
}

template <const bundle_lengths lengths>
struct contraction_indices {
  consteval contraction_indices(const contraction_labels& labels) {
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

    for (size_t i = 0; i < contracted.size(); ++i) {
      AP[i] = labels.a.find(contracted[i]);
      BP[i] = labels.b.find(contracted[i]);
    }

    for (size_t i = 0; i < free_a.size(); ++i) {
      AI[i] = labels.a.find(free_a[i]);
      CI[i] = labels.c.find(free_a[i]);
    }

    for (size_t i = 0; i < free_b.size(); ++i) {
      BJ[i] = labels.b.find(free_b[i]);
      CJ[i] = labels.c.find(free_b[i]);
    }
  }

  std::array<std::size_t, lengths.I> CI{};
  std::array<std::size_t, lengths.J> CJ{};

  std::array<std::size_t, lengths.I> AI{};
  std::array<std::size_t, lengths.P> AP{};

  std::array<std::size_t, lengths.P> BP{};
  std::array<std::size_t, lengths.J> BJ{};
};

enum memory_layout { ROW_MAJOR, COL_MAJOR };

template <typename T, std::size_t ndim>
struct tensor {
  std::array<std::size_t, ndim> dimensions;
  T* data;
  memory_layout _layout;
};

template <std::size_t ndim>
consteval std::array<std::size_t, ndim> compute_strides(
    std::array<std::size_t, ndim> dimensions, memory_layout layout) {
  std::array<std::size_t, ndim> strides;
  switch (layout) {
    case ROW_MAJOR:
      throw std::runtime_error("not implemented yet.");
    case COL_MAJOR:
      strides[0] = 1;
      for (std::size_t i = 1; i < ndim; ++i) {
        strides[i] = strides[i - 1] * dimensions[i - 1];
      }
  }
  return strides;
}

template <std::size_t ndim, std::size_t nindices>
struct scatter_vector_info {
  consteval scatter_vector_info(const std::array<std::size_t, nindices> indices,
                                const std::array<std::size_t, ndim> dimensions,
                                const std::array<std::size_t, ndim> strides) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
      const std::size_t& idx = indices[i];
      lengths[i] = dimensions[idx];
      strds[i] = strides[idx];
    }
  }

  /// @brief the length for each of the dimensions.
  std::array<std::size_t, nindices> lengths;

  /// @brief the stride for each of the dimensions.
  std::array<std::size_t, nindices> strds;
};

template <std::size_t ndim, std::size_t nindices>
std::vector<std::size_t> compute_scatter_vector(
    const scatter_vector_info<nindices, ndim>& info) {

  // compute scatter vectors.
  // first, create 2D vector with each index, stride combination.
  std::vector<std::vector<std::size_t>> parts{};
  for (std::size_t i = 0; i < info.lengths.size(); ++i) {
    const std::size_t& dim = info.lengths[i];
    const std::size_t& stride = info.strds[i];

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

  return std::reduce(parts.begin() + 1, parts.end(), parts[0], elementwise_add);
}

template <std::size_t b>
struct block_scatter {
  const std::vector<std::size_t>& scat;

  std::vector<std::size_t> operator()() const {
    const size_t nblocks = std::ceil(scat.size() / b);

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
