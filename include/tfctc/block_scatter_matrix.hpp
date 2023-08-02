#pragma once

#include "tensor.hpp"
#include "scatter_matrix.hpp"
#include "block_scatter_vector.hpp"
#include "utils.hpp"

namespace tfctc
{
  namespace internal
  {
    template <typename T>
    class BlockScatterMatrix : public ScatterMatrix<T>
    {
    public:
      BlockScatterMatrix(Tensor<T>& t, std::vector<size_t> row_indices, std::vector<size_t> col_indices, size_t br, size_t bc)
        : ScatterMatrix<T>(t, row_indices, col_indices),
        rbs(this->rscat, br),
        cbs(this->cscat, bc) {}

      const size_t row_stride_in_block(size_t i)
      {
        return this->rbs.at(i);
      }

      const size_t col_stride_in_block(size_t j)
      {
        return this->cbs.at(j);
      }
    private:
      BlockScatterVector rbs;
      BlockScatterVector cbs;
    };
  };
};