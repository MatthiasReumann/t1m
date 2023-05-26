#pragma once

#include <vector>
#include "scat.h"
#include "marray.hpp"

class ScatterVector
{
public:
  ScatterVector(MArray::len_vector lengths, MArray::len_vector strides, std::vector<size_t> indices)
  {
    std::vector<size_t> l, s;
    for (auto &idx : indices)
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
private:
  std::vector<size_t> scat;
};