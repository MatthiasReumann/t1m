#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <unordered_set>
#include <vector>

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
};  // namespace t1m::internal
