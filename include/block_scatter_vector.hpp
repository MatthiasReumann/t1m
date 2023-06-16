#pragma once

#include <vector>
#include "scatter_vector.hpp"

namespace tfctc
{
  namespace internal
  {
    class BlockScatterVector
    {
    public:
      BlockScatterVector(ScatterVector& scat, size_t b)
      {
        size_t stride; // s, if constant; 0, if different strides
        const size_t l = scat.size();
        const size_t size = std::ceil(l / static_cast<float>(b)); // ⌈l/b⌉

        this->bs.reserve(size);

        for (int i = 0; i < l; i += b) // blocks
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
        }
      }

      size_t at(size_t i)
      {
        return this->bs.at(i);
      }
      std::vector<size_t> bs;
    };
  };
};
