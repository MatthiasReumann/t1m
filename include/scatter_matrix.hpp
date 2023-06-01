#pragma once

#include "tensor.hpp"
#include "scatter_vector.hpp"
#include "block_scatter_vector.hpp"
#include "utils.hpp"
#include "blis.h"
#include <vector>

template <typename T>
class ScatterMatrix : public Tensor<T>
{
public:
  ScatterMatrix(Tensor<T> &t, std::vector<size_t> row_indices, std::vector<size_t> col_indices)
      : Tensor<T>(t),
        rscat(this->lengths(), this->strides(), row_indices),
        cscat(this->lengths(), this->strides(), col_indices),
        rbs(this->rscat),
        cbs(this->cscat) {}

  template <typename U>
  void pack_to_cont_buffer_col(U *buffer, int off_i, int off_j, dim_t m, dim_t n, dim_t mr)
  {
    const U *ptr = this->cdata();

    for (int i = 0; i < m; i += mr)
    {
      for (int j = 0; j < n; j++)
      {
        for (int k = 0; k < mr; k++)
        {
          const int x = k + off_i + i;
          const int y = j + off_j;

          if (x < m)
          {
            buffer[i * n + k + j * mr] = ptr[this->location(x, y)];
          }
        }
      }
    }
  }

  template <typename U>
  /*
  OUTPUT: Layout of parameter *buffer
     <--nr-->
     <------ n ------->
  ^   ------------------
  |  | --->-⌄ | ----->  |
  |  | >----⌄ | /---->  |
  |  | >----⌄ | /---->  |
  m  | >----⌄ | /---->  |
  |  | >----⌄ | /---->  |
  |  | >----⌄ | /---->  |
  |  | >----- | /---->  |
  ⌄   ------------------
  */
  void pack_to_cont_buffer_row(U *buffer, int off_i, int off_j, dim_t m, dim_t n, dim_t nr)
  {
    const U *ptr = this->cdata();

    for (int j = 0; j < n; j += nr)
    {
      for (int i = 0; i < m; i++)
      {
        for (int k = 0; k < nr; k++)
        {
          const int x = i + off_i;
          const int y = k + off_j + j;
          if (y < n)
          {
            buffer[j * m + k + i * nr] = ptr[this->location(x, y)];
          }
        }
      }
    }
  }

  template <typename U>
  void unpack_from_buffer(U *buffer, int off_i, int off_j, dim_t m, dim_t n)
  {
    T *ptr = this->data();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        ptr[this->location(i + off_i, j + off_j)] += buffer[i + j * m];
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