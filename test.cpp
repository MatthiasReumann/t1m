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
    print_vec(this->scat);
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
        cscat(this->lengths(), this->strides(), col_indices) {}

  template <typename T, int m, int n> // m x n
  void pack_to_submatrix(T* submatrix)
  {
    const T* ptr = this->cdata();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        submatrix[i + j * m] = ptr[this->location(i, j)];
      }
    }
  }

  template <typename T, int m, int n> // m x n
  void pack_from_submatrix(T* submatrix)
  {
    T* ptr = this->data();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        ptr[this->location(i, j)] = submatrix[i + j * m];
      }
    }
  }

  int location(int i, int j)
  {
    return this->rscat.at(i) + this->cscat.at(j);
  }

  std::vector<size_t> get_rscat()
  {
    return this->rscat.scat;
  }

  std::vector<size_t> get_cscat()
  {
    return this->cscat.scat;
  }

  ScatterVector rscat;
  ScatterVector cscat;
};

template <int mc, int nc, int kc>
void gemm(ScatterMatrix *A, ScatterMatrix *B, ScatterMatrix *C)
{
  // B_(PJ)
  auto rscat_I_C = C->get_rscat();
  auto cscat_J_C = C->get_cscat();

  // A_(IP)
  auto rscat_I_A = A->get_rscat();
  auto cscat_P_A = A->get_cscat();

  // B_(PJ)
  auto rscat_P_B = B->get_rscat();
  auto cscat_J_B = B->get_cscat();

  size_t m = rscat_I_A.size();
  size_t k = cscat_P_A.size();
  size_t n = cscat_J_B.size();

  /*
  TODO: Use microkernel only for full matrices. Use normal GEMM as
  obove for the last (not full) matrices.
  */

  // TODO: Rename
  /*
  int frac_I = (int)(m / MC);
  int frac_P = (int)(k / KC);
  int frac_J = (int)(n / NC);

  int rest_I = (int)(m % MC);
  int rest_P = (int)(k % KC);
  int rest_J = (int)(n % NC);

  float *A_ = A->data();
  float *B_ = B->data();
  float *C_ = C->data();

  std::cout << frac_I << std::endl;
  std::cout << frac_P << std::endl;
  std::cout << frac_J << std::endl;

  std::cout << rest_I << std::endl;
  std::cout << rest_P << std::endl;
  std::cout << rest_J << std::endl; */

  float *A_tilde = nullptr; // A in R^{mc x kc}
  float *B_tilde = nullptr; // B in G^{kc x nc}
  float *C_tilde = nullptr; // B in G^{mc x nc}
  alloc_aligned<float>(&A_tilde, mc * kc);
  alloc_aligned<float>(&B_tilde, kc * nc);
  alloc_aligned<float>(&C_tilde, mc * nc);

  for (int j_c = 0; j_c < n; j_c += nc)
  {
    for (int p_c = 0; p_c < k; p_c += kc)
    {
      B->pack_to_submatrix<float, kc, nc>(B_tilde);
      for (int i_c = 0; i_c < m; i_c += mc)
      {
        A->pack_to_submatrix<float, mc, kc>(A_tilde);
        
        macrokernel_simple<mc, nc, kc>(A_tilde, B_tilde, C_tilde);

        C->pack_from_submatrix<float, mc, nc>(C_tilde);
      }
    }
  }

  /*
  for (int i = frac_I * MC; i < rscat_I_A.size(); i++)
  {
    for (int j = frac_J * NC; j < cscat_J_B.size(); j++)
    {
      float c_ij = 0.;
      for (int p = frac_P * KC; p < cscat_P_A.size(); p++)
      {
        const float a = A_[rscat_I_A[i] + cscat_P_A[p]];
        const float b = B_[rscat_P_B[p] + cscat_J_B[j]];
        c_ij += a * b;
      }

      C_[rscat_I_C[i] + cscat_J_C[j]] = c_ij;
    }
  }*/
}

void contract(Tensor<float> A, std::string labelsA,
              Tensor<float> B, std::string labelsB,
              Tensor<float> C, std::string labelsC)
{
  auto indexLabelFinder = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto scatterA = new ScatterMatrix(A, indexLabelFinder->I, indexLabelFinder->Pa);
  auto scatterB = new ScatterMatrix(B, indexLabelFinder->Pb, indexLabelFinder->J);
  auto scatterC = new ScatterMatrix(C, indexLabelFinder->Ic, indexLabelFinder->Jc);

  // B_(PJ)
  auto rscat_I_C = scatterC->get_rscat();
  auto cscat_J_C = scatterC->get_cscat();

  // A_(IP)
  auto rscat_I_A = scatterA->get_rscat();
  auto cscat_P_A = scatterA->get_cscat();

  // B_(PJ)
  auto rscat_P_B = scatterB->get_rscat();
  auto cscat_J_B = scatterB->get_cscat();

  gemm<2, 2, 2>(scatterA, scatterB, scatterC);
}

void test()
{
  float *A_ptr = nullptr;
  float *B_ptr = nullptr;
  float *C_ptr = nullptr;

  alloc_aligned<float>(&A_ptr, 2 * 2);
  alloc_aligned<float>(&B_ptr, 2 * 2);
  alloc_aligned<float>(&C_ptr, 2 * 2);

  A_ptr[0] = 3.;
  A_ptr[2] = 1.;
  A_ptr[1] = 4.;
  A_ptr[3] = 7.;

  B_ptr[0] = 1.;
  B_ptr[2] = 0.;
  B_ptr[1] = 1.;
  B_ptr[3] = 1.;

  auto A_lengths = {2, 2};
  auto B_lengths = {2, 2};
  auto C_lengths = {2, 2};

  std::cout << "sizeof(float) = " << sizeof(float) << std::endl;

  auto A = Tensor<float>(A_lengths, A_ptr, MArray::COLUMN_MAJOR);
  auto B = Tensor<float>(B_lengths, B_ptr, MArray::COLUMN_MAJOR);
  auto C = Tensor<float>(C_lengths, C_ptr, MArray::COLUMN_MAJOR);

  contract(A, "ba", B, "bc", C, "ca");

  std::cout << A << std::endl;
  std::cout << B << std::endl;
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
  test();
  // test_macrokernel_simple();

  return 1;
}