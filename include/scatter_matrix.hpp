#pragma once

#include "definitions.hpp"
#include "scatter_vector.hpp"
#include "block_scatter_vector.hpp"
#include "utils.hpp"
#include <vector>

class ScatterMatrix : public Tensor<float>
{
public:
  ScatterMatrix(Tensor<float> &t, std::vector<size_t> row_indices, std::vector<size_t> col_indices)
      : Tensor<float>(t),
        rscat(this->lengths(), this->strides(), row_indices),
        cscat(this->lengths(), this->strides(), col_indices),
        rbs(this->rscat),
        cbs(this->cscat) {}

  template <typename T, int m, int n> // m x n
  void pack_to_cont_buffer_col(T *buffer, int off_i, int off_j)
  {
    const T *ptr = this->cdata();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        buffer[i + j * m] = ptr[this->location(i + off_i, j + off_j)];
      }
    }
  }

  template <typename T, int m, int n> // m x n
  void pack_to_cont_buffer_row(T *buffer, int off_i, int off_j)
  {
    const T *ptr = this->cdata();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        buffer[i + j * m] = ptr[this->location(i + off_i, j + off_j)];
      }
    }
  }

  template <typename T, int m, int n> // m x n
  void add_from_submatrix(T *submatrix, int off_i, int off_j)
  {
    T *ptr = this->data();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        ptr[this->location(i + off_i, j + off_j)] += submatrix[i + j * m];
      }
    }
  }

  int row_size()
  {
    return this->rscat.size();
  }

  int col_size()
  {
    return this->cscat.size();
  }

  int location(int i, int j)
  {
    return this->rscat.at(i) + this->cscat.at(j);
  }
private:
  ScatterVector rscat;
  ScatterVector cscat;

  BlockScatterVector<4> rbs;
  BlockScatterVector<4> cbs;
};