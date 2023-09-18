#pragma once

#include "tensor.hpp"
#include "scatter_matrix.hpp"
#include "block_scatter_vector.hpp"
#include "utils.hpp"

namespace t1m::internal
{
  template <typename T>
  class BlockScatterMatrix : public ScatterMatrix<T>
  {
  public:
    BlockScatterMatrix(Tensor<T>& t, std::vector<size_t> row_indices, std::vector<size_t> col_indices, size_t br, size_t bc)
      : ScatterMatrix<T>(t, row_indices, col_indices) {
        this->rbs = t1m::internal::calc_block_scatter(this->rscat.scat, br);
        this->cbs = t1m::internal::calc_block_scatter(this->cscat.scat, bc);
      }

    size_t row_stride_in_block(size_t i) const
    {
      return this->rbs.at(i);
    }

    size_t col_stride_in_block(size_t j) const
    {
      return this->cbs.at(j);
    }

  private:
    std::vector<size_t> rbs;
    std::vector<size_t> cbs;
  };
};