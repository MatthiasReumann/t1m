#pragma once

#include <vector>
#include "scatter_vector.hpp"
#include "utils.hpp"

namespace tfctc
{
  namespace internal
  {
    class BlockScatterVector
    {
    public:
      BlockScatterVector(ScatterVector& scat, size_t b)
      {
        size_t i, stride; // s, if constant; 0, if different strides
        const size_t l = scat.size();
        const size_t size = std::ceil(l / static_cast<float>(b)); // ⌈l/b⌉

        this->bs.reserve(size);

        for (i = 0; i < size_t(l / b) * b; i += b) // blocks
        {
          stride = scat.at(i + 1) - scat.at(i);

          for (size_t j = i + 2; j < i + b; j++) {
            if ((scat.at(j) - scat.at(j - 1)) != stride)
            {
              stride = 0;
              break;
            }
          }
          this->bs.push_back(stride);
        }

        stride = 0;
        if (i + 1 < l)
        {
          stride = scat.at(i + 1) - scat.at(i);

          for (size_t j = i + 2; j < l; j++) {
            if ((scat.at(j) - scat.at(j - 1)) != stride)
            {
              stride = 0;
              break;
            }

          }
        }
        this->bs.push_back(stride);
      }

      size_t at(size_t i)
      {
        return this->bs.at(i);
      }
    private:
      std::vector<size_t> bs;
    };
  };
};
