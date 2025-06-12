#pragma once

#include <algorithm>
#include <string>
#include <iterator>
#include <vector>

namespace t1m {
namespace internal {
struct index_bundles {
  std::vector<std::size_t> CI;
  std::vector<std::size_t> CJ;

  std::vector<std::size_t> AI;
  std::vector<std::size_t> AP;

  std::vector<std::size_t> BP;
  std::vector<std::size_t> BJ;
};

inline index_bundles get_index_bundles(const std::string& labels_a,
                                const std::string& labels_b,
                                const std::string& labels_c) {
  index_bundles bundles{};

  // Copy for sorting.
  std::string a(labels_a);
  std::string b(labels_b);
  std::string c(labels_c);

  // Sort for set operations.
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::sort(c.begin(), c.end());

  // Apply set operations.
  std::string contracted, free_a, free_b;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(contracted));
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::back_inserter(free_a));
  std::set_difference(b.begin(), b.end(), a.begin(), a.end(),
                      std::back_inserter(free_b));

  // Assign bundles.
  bundles.AP.reserve(contracted.size());
  bundles.BP.reserve(contracted.size());
  for (size_t i = 0; i < contracted.size(); ++i) {
    bundles.AP.push_back(labels_a.find(contracted[i]));
    bundles.BP.push_back(labels_b.find(contracted[i]));
  }

  bundles.AI.reserve(free_a.size());
  bundles.CI.reserve(free_a.size());
  for (size_t i = 0; i < free_a.size(); ++i) {
    bundles.AI.push_back(labels_a.find(free_a[i]));
    bundles.CI.push_back(labels_c.find(free_a[i]));
  }

  bundles.BJ.reserve(free_b.size());
  bundles.CJ.reserve(free_b.size());
  for (size_t i = 0; i < free_b.size(); ++i) {
    bundles.BJ.push_back(labels_b.find(free_b[i]));
    bundles.CJ.push_back(labels_c.find(free_b[i]));
  }

  return bundles;
}
};  // namespace internal
};  // namespace t1m
