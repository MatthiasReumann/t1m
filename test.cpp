#include <iostream>
#include "utils.h"
#include "index_bundle_finder.h"
#include "scat.h"
#include "macrokernel.h"
#include "marray.hpp"

constexpr int MR = 6;
constexpr int NR = 8;
constexpr int KP = 4;
constexpr int MC = 72;
constexpr int NC = 4080;
constexpr int KC = 256;

template <class T>
using Tensor = MArray::marray_view<T>;

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

  std::vector<size_t> scat;
};

class ScatterMatrix : public Tensor<float>
{
public:
  ScatterMatrix(Tensor<float> &t, std::vector<size_t> row_indices, std::vector<size_t> col_indices)
      : Tensor<float>(t),
        rscat(this->lengths(), this->strides(), row_indices),
        cscat(this->lengths(), this->strides(), col_indices) { }

  template <typename T, int m, int n> // m x n
  void pack_to_submatrix(T *submatrix, int off_i, int off_j)
  {
    const T *ptr = this->cdata();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        submatrix[i + j * m] = ptr[this->location(i + off_i, j + off_j)];
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

  ScatterVector rscat;
  ScatterVector cscat;
};

template <int mc, int nc, int kc>
void gemm(ScatterMatrix *A, ScatterMatrix *B, ScatterMatrix *C)
{
  float *A_ = A->data();
  float *B_ = B->data();
  float *C_ = C->data();

  size_t m = A->row_size();
  size_t k = A->col_size();
  size_t n = B->col_size();

  float *A_tilde = nullptr; // A in R^{mc x kc}
  float *B_tilde = nullptr; // B in G^{kc x nc}
  float *C_tilde = nullptr; // C in G^{mc x nc}

  alloc_aligned<float>(&A_tilde, mc * kc);
  alloc_aligned<float>(&B_tilde, kc * nc);
  alloc_aligned<float>(&C_tilde, mc * nc);

  for (int j_c = 0; j_c < int(n/nc); j_c++)
  {
    for (int p_c = 0; p_c < int(k/kc); p_c++)
    {
      B->pack_to_submatrix<float, kc, nc>(B_tilde,  p_c * kc, j_c * nc);
      
      for (int i_c = 0; i_c < int(m/mc); i_c++)
      {
        A->pack_to_submatrix<float, mc, kc>(A_tilde, i_c * mc, p_c * kc);

        macrokernel_simple<mc, nc, kc>(A_tilde, B_tilde, C_tilde);

        C->add_from_submatrix<float, mc, nc>(C_tilde, i_c * mc, j_c * nc);
      }
    }
  }

  for (int i = (int(m/mc) * mc); i < m; i++)
  {
    for (int j = (int(n/nc) * nc); j < n; j++)
    {
      float c_ij = 0.;
      for (int p = (int(k/kc) * kc); p < k; p++)
      {
        A->location(i, p);
        const float a = A_[A->location(i, p)];
        const float b = B_[B->location(p, j)];
        c_ij += a * b;
      }

      C_[C->location(i, j)] = c_ij;
    }
  }

  free(A_tilde);
  free(B_tilde);
  free(C_tilde);
}

void contract(Tensor<float> A, std::string labelsA,
              Tensor<float> B, std::string labelsB,
              Tensor<float> C, std::string labelsC)
{
  auto indexLabelFinder = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto scatterA = new ScatterMatrix(A, indexLabelFinder->I, indexLabelFinder->Pa);
  auto scatterB = new ScatterMatrix(B, indexLabelFinder->Pb, indexLabelFinder->J);
  auto scatterC = new ScatterMatrix(C, indexLabelFinder->Ic, indexLabelFinder->Jc);

  gemm<5, 5, 5>(scatterA, scatterB, scatterC);
}

void test4x4()
{
  std::cout << "TEST 4x4 . 4x4 = 4x4" << '\n';
  float *A_ptr = nullptr;
  float *B_ptr = nullptr;
  float *C_ptr = nullptr;

  alloc_aligned<float>(&A_ptr, 4 * 4);
  alloc_aligned<float>(&B_ptr, 4 * 4);
  alloc_aligned<float>(&C_ptr, 4 * 4);

  A_ptr[0] = 3.; A_ptr[4] = 1.; A_ptr[8] = 0.;  A_ptr[12] =  1.;
  A_ptr[1] = 4.; A_ptr[5] = 7.; A_ptr[9] = 1.1; A_ptr[13] = 1.;
  A_ptr[2] = 1.; A_ptr[6] = 0.; A_ptr[10] = 1.; A_ptr[14] = 1.;
  A_ptr[3] = 1.; A_ptr[7] = 0.; A_ptr[11] = 0.; A_ptr[15] = 1.;

  B_ptr[0] = 1.; B_ptr[4] = 1.; B_ptr[8] = 0.;   B_ptr[12] = 1.; 
  B_ptr[2] = 0.; B_ptr[5] = 0.; B_ptr[9] = 1.;   B_ptr[13] = 1.7;
  B_ptr[1] = 1.; B_ptr[6] = 1.; B_ptr[10] = 4.3; B_ptr[14] = 2.;
  B_ptr[3] = 1.; B_ptr[7] = 3.; B_ptr[11] = 1.;  B_ptr[15] = 1.;

  auto A_lengths = {4, 4};
  auto B_lengths = {4, 4};
  auto C_lengths = {4, 4};

  auto A = Tensor<float>(A_lengths, A_ptr, MArray::COLUMN_MAJOR);
  auto B = Tensor<float>(B_lengths, B_ptr, MArray::COLUMN_MAJOR);
  auto C = Tensor<float>(C_lengths, C_ptr, MArray::COLUMN_MAJOR);

  std::cout << A << std::endl;
  std::cout << B << std::endl;

  contract(A, "ba", B, "bc", C, "ca");

  std::cout << C << std::endl;
}

void test4x3()
{
  std::cout << "TEST 4x3 . 3x4" << '\n';
  float *A_ptr = nullptr;
  float *B_ptr = nullptr;
  float *C_ptr = nullptr;

  alloc_aligned<float>(&A_ptr, 4 * 3);
  alloc_aligned<float>(&B_ptr, 4 * 3);
  alloc_aligned<float>(&C_ptr, 3 * 3);

  A_ptr[0] = 3.; A_ptr[4] = 1.; A_ptr[8] = 0.;  
  A_ptr[1] = 4.; A_ptr[5] = 7.; A_ptr[9] = 1.1;
  A_ptr[2] = 1.; A_ptr[6] = 0.; A_ptr[10] = 1.; 
  A_ptr[3] = 1.; A_ptr[7] = 0.; A_ptr[11] = 0.;

  B_ptr[0] = 1.; B_ptr[4] = 1.; B_ptr[8] = 0.;  
  B_ptr[2] = 0.; B_ptr[5] = 0.; B_ptr[9] = 1.; 
  B_ptr[1] = 1.; B_ptr[6] = 1.; B_ptr[10] = 4.3;
  B_ptr[3] = 1.; B_ptr[7] = 3.; B_ptr[11] = 1.;

  auto A_lengths = {4, 3};
  auto B_lengths = {4, 3};
  auto C_lengths = {3, 3};

  auto A = Tensor<float>(A_lengths, A_ptr, MArray::COLUMN_MAJOR);
  auto B = Tensor<float>(B_lengths, B_ptr, MArray::COLUMN_MAJOR);
  auto C = Tensor<float>(C_lengths, C_ptr, MArray::COLUMN_MAJOR);

  std::cout << A << std::endl;
  std::cout << B << std::endl;

  contract(A, "ba", B, "bc", C, "ca");

  std::cout << C << std::endl;
}

void test2()
{
  auto A_lengths = {2, 4, 3, 3};
  auto B_lengths = {4, 4, 6};
  auto C_lengths = {6, 3, 2, 3, 4};

  float *A_ptr = nullptr;
  float *B_ptr = nullptr;
  float *C_ptr = nullptr;

  alloc_aligned<float>(&A_ptr, 2 * 4 * 3 * 3);
  alloc_aligned<float>(&B_ptr, 4 * 4 * 6);
  alloc_aligned<float>(&C_ptr, 6 * 3 * 2 * 3 * 4);

  std::cout << "sizeof(float) = " << sizeof(float) << std::endl;

  auto A = Tensor<float>(A_lengths, A_ptr, MArray::COLUMN_MAJOR);
  auto B = Tensor<float>(B_lengths, B_ptr, MArray::COLUMN_MAJOR);
  auto C = Tensor<float>(C_lengths, C_ptr, MArray::COLUMN_MAJOR);

  // C_abcde = A_cfbd . B_fea
  contract(A, "cfbd", B, "fea", C, "abcde");

  /*
  std::cout << A << std::endl;
  std::cout << B << std::endl;
  std::cout << C << std::endl;*/
}

void test_macrokernel_simple()
{
  float *A, *B, *C;

  alloc_aligned<float>(&A, 2 * 2);
  alloc_aligned<float>(&B, 2 * 2);
  alloc_aligned<float>(&C, 2 * 2);

  A[0] = 1.;
  A[2] = 2.7;
  A[1] = 3.;
  A[3] = 1.0;
  B[0] = 1.;
  B[2] = 1.;
  B[1] = 2.;
  B[3] = 0.;

  macrokernel_simple<2, 2, 2>(A, B, C);

  print_mat(C, 2, 2);
}

int main()
{
  test4x4();
  test4x3();
  // test_macrokernel_simple();

  return 1;
}