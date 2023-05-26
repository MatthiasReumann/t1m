#pragma once

#include <vector>
#include "scatter_vector.hpp"

template <size_t b>
class BlockScatterVector
{
public:
  BlockScatterVector(ScatterVector &scat)
  {
    bool isConst;
    size_t d1, d2;
    const auto size = scat.size() / b;

    this->bs.reserve(size);
    std::cout << size << std::endl;

    for (int i = 1; i < size; i += b)
    {
      isConst = true;
      d1 = scat.at(i) - scat.at(i - 1);
      for (int j = i + 1; j < i * b; j++)
      {
        std::cout << "hello" << std::endl;
        d2 = scat.at(i) - scat.at(i - 1);
        if (d2 != d1)
        {
          isConst = false;
          break;
        }
        d1 = scat.at(i) - scat.at(i - 1);
      }
      this->bs.push_back(isConst);
    }
  }

  std::vector<bool> bs;
};