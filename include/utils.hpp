#pragma once

#include <iostream>

template <typename T>
void alloc_aligned(T **ptr, size_t n)
{
  // TODO: Memory Alignment?
  if (posix_memalign((void **)ptr, 32, n * sizeof(T)))
  {
    std::throw_with_nested(std::bad_alloc());
  }
}

template <typename T>
void print_vec(std::vector<T> &vec)
{
  std::cout << "[ ";
  for (auto v : vec)
  {
    std::cout << v << " ";
  }
  std::cout << ']' << '\n';
}

template <typename T>
void print_linear(T *mat, int m, int n)
{
  for (int i = 0; i < m * n - 1; i++)
  {
    std::cout << mat[i] << " ";
  }
  std::cout << mat[m * n - 1] << '\n';
}

template <typename T>
void print_mat(T *mat, int rows, int columns)
{
  std::cout << '{' << '\n';
  for (int i = 0; i < rows; i++)
  {
    std::cout << ' ' << '{';
    for (int j = 0; j < columns - 1; j++)
    {
      std::cout << mat[i + j * rows] << ", ";
    }
    std::cout << mat[i + (columns - 1) * rows];
    std::cout << '}' << ',' << '\n';
  }
  std::cout << '}' << '\n';
}

template <typename T>
void print_mat_row(T *mat, int rows, int columns)
{
  std::cout << '{' << '\n';
  for (int i = 0; i < rows; i++)
  {
    std::cout << ' ' << '{';
    for (int j = 0; j < columns - 1; j++)
    {
      std::cout << mat[j + i * columns] << ", ";
    }
    std::cout << mat[columns - 1 + i * rows];
    std::cout << '}' << ',' << '\n';
  }
  std::cout << '}' << '\n';
}