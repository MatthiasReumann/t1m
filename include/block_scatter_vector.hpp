#pragma once

#include <vector>
#include "scatter_vector.hpp"

template <size_t b>
class BlockScatterVector
{
public:
  BlockScatterVector(ScatterVector &scat)
  {
    /*size_t stride; // s, if constant; 0, if different strides
    const auto l = scat.size();
    const auto size = std::ceil(l / static_cast<float>(b)); // ⌈l/b⌉

    this->bs.reserve(size);

    for (int i = 0; i < scat.size(); i += b) // blocks
    {
      stride = scat.at(i + 1) - scat.at(i);
      for (int j = i; j < std::min(i + b, l) - 1; j++)
      {
        if (stride != scat.at(j + 1) - scat.at(j))
        {
          stride = 0; break;
        }
      }
      this->bs.push_back(stride);
    }*/
  }

  std::vector<size_t> bs;
};
