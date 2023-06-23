#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include "marray.hpp"

namespace tfctc
{
  namespace internal
  {
    std::vector<size_t> get_scat(std::vector<size_t> sizes, std::vector<size_t> strides);

    class ScatterVector
    {
    public:
      ScatterVector(MArray::len_vector lengths, MArray::len_vector strides, std::vector<size_t> indices)
      {
        std::vector<size_t> l, s;
        for (auto& idx : indices)
        {
          l.push_back(static_cast<size_t>(lengths.at(idx)));
          s.push_back(static_cast<size_t>(strides.at(idx)));
        }
        this->scat = get_scat(l, s);
      }

      const size_t size()
      {
        return this->scat.size();
      }

      const size_t at(int i)
      {
        return this->scat.at(i);
      }

      std::vector<size_t> scat;
    };

    std::vector<size_t> one_to_many(std::vector<size_t> values, size_t summand)
    {
      for (int i = 0; i < values.size(); i++)
        values[i] += summand;
      return values;
    }

    std::vector<std::vector<size_t>> many_to_many(std::vector<size_t> values1, std::vector<size_t> values2)
    {
      std::vector<std::vector<size_t>> res;
      res.reserve(values1.size());

      for (auto& summand : values1)
        res.push_back(one_to_many(values2, summand));

      return res;
    }

    std::vector<std::vector<size_t>> multi_level(std::vector<std::vector<size_t>> levels)
    {
      if (levels.size() == 1)
      {
        return levels;
      }

      auto lv1 = levels[0];
      auto lv2 = levels[1];

      auto combined = many_to_many(lv1, lv2);

      std::vector<std::vector<size_t>> l;
      for (auto& comb_it : combined)
      {
        auto minus2 = std::vector<std::vector<size_t>>{ {comb_it} };
        minus2.insert(minus2.end(), levels.begin() + 2, levels.end());
        auto ml = multi_level(minus2);
        l.insert(l.end(), ml.begin(), ml.end());
      }

      return l;
    }

    std::vector<size_t> get_scat(std::vector<size_t> sizes, std::vector<size_t> strides)
    {
      assert(sizes.size() == strides.size());

      const size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<size_t>());

      std::vector<size_t> scat_vec;
      std::vector<std::vector<size_t>> levels;
      scat_vec.reserve(total_size);
      levels.reserve(sizes.size());

      // pre calcute i * stride_i for each i
      for (int i = 0; i < sizes.size(); i++)
      {
        levels.push_back({});
        for (int j = 0; j < sizes[i]; j++)
        {
          levels[i].push_back(j * strides[i]);
        }
      }

      std::reverse(levels.begin(), levels.end());
      auto ml = multi_level(levels);

      // flatten
      for (auto& x : ml)
      {
        for (auto& y : x)
        {
          scat_vec.push_back(y);
        }
      }

      return scat_vec;
    }

  };
};