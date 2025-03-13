#pragma once

#include "tensor.hpp"
#include "scatter.hpp"
#include <vector>

namespace t1m::utils
{
  template <typename T>
  class ScatterMatrix : public Tensor<T>
  {
  public:
    ScatterMatrix(Tensor<T>& t, std::vector<size_t> row_indices, std::vector<size_t> col_indices)
      : Tensor<T>(t),
      rscat(this->calc_scatter(row_indices)),
      cscat(this->calc_scatter(col_indices)) {}

    T operator() (size_t i, size_t j) const
    {
      return this->get(i, j);
    }

    T get(size_t i, size_t j) const
    {
      return this->cdata()[this->location(i, j)];
    }

    T* pointer_at_loc(size_t i, size_t j) const
    {
      return &(this->data()[this->location(i, j)]);
    }

    size_t row_size() const
    {
      return this->rscat.size();
    }

    size_t col_size() const
    {
      return this->cscat.size();
    }

    size_t location(size_t i, size_t j) const
    {
      return this->rscat.at(i) + this->cscat.at(j);
    }
  protected:
    std::vector<size_t> rscat;
    std::vector<size_t> cscat;
  private:
    std::vector<size_t> calc_scatter(std::vector<size_t> indices) {
      std::vector<size_t> lengths, strides;

      for (const auto& idx : indices)
      {
        lengths.push_back(static_cast<size_t>(this->lengths().at(idx)));
        strides.push_back(static_cast<size_t>(this->strides().at(idx)));
      }

      return t1m::utils::calc_scatter(lengths, strides);
    }
  };
};