#include <iostream>
#include <unordered_set>
#include "scat.h"
#include "marray.hpp"

template <typename T>
void print_vec(std::vector<T> &vec)
{
  std::cout << "[ ";
  for (auto &v : vec)
  {
    std::cout << v << " ";
  }
  std::cout << ']' << '\n';
}

class IndexBundleFinder
{
public:
  IndexBundleFinder(std::string labelsA, std::string labelsB, std::string labelsC)
      : labelsA(labelsA), labelsB(labelsB), labelsC(labelsC)
  {
    this->find();
    this->find_c_permutation();
  }

  void find()
  {
    bool in_I;
    std::unordered_set<char> setB{labelsB.cbegin(), labelsB.cend()};

    for (size_t i = 0; i < labelsA.length(); i++)
    {
      in_I = false;
      for (size_t j = 0; j < labelsB.length(); j++)
      {
        if (labelsA.at(i) == labelsB.at(j))
        {
          in_I = true;
          this->Pa.push_back(i);
          this->Pb.push_back(j);
          setB.erase(labelsA.at(i));
        }
      }

      if (!in_I)
        this->I.push_back(i);
    }

    for (int j = 0; j < labelsB.length(); j++)
      if (setB.count(labelsB.at(j)) > 0)
      {
        this->J.push_back(j);
      }
  }

  void print()
  {
    std::cout << "I:" << std::endl;
    print_vec(this->I);
    std::cout << "J:" << std::endl;
    print_vec(this->J);
    std::cout << "Pa:" << std::endl;
    print_vec(this->Pa);
    std::cout << "Pb:" << std::endl;
    print_vec(this->Pb);

    std::cout << "Ic:" << std::endl;
    print_vec(this->Ic);

    std::cout << "Jc:" << std::endl;
    print_vec(this->Jc);
  }

  void find_c_permutation()
  {
    for (auto &idx : this->I)
    {
      for (int j = 0; j < this->labelsC.length(); j++)
      {
        if(this->labelsA.at(idx) == this->labelsC.at(j)) {
          this->Ic.push_back(j);
        }
      }
    }

    for (auto &idx : this->J)
    {
      for (int j = 0; j < this->labelsC.length(); j++)
      {
        if(this->labelsB.at(idx) == this->labelsC.at(j)) {
          this->Jc.push_back(j);
        }
      }
    }
  }

  std::vector<size_t> I;
  std::vector<size_t> J;
  std::vector<size_t> Pa;
  std::vector<size_t> Pb;

  std::vector<size_t> Ic;
  std::vector<size_t> Jc;

private:
  std::string labelsA;
  std::string labelsB;
  std::string labelsC;
};

template <class T>
using Tensor = MArray::marray_view<T>;

class TensorMatrix : public Tensor<float>
{
public:
  // TODO: Naming or proper templating to differentiate between A and B
  TensorMatrix(Tensor<float> &t, std::vector<size_t> row_indices, std::vector<size_t> col_indices)
      : Tensor<float>(t)
  {
    std::vector<size_t> row_lengths, row_strides, col_lengths, col_strides;
    for (auto &idx : row_indices)
    {
      row_lengths.push_back(static_cast<size_t>(this->length(idx)));
      row_strides.push_back(static_cast<size_t>(this->stride(idx)));
    }
    for (auto &idx : col_indices)
    {
      col_lengths.push_back(static_cast<size_t>(this->length(idx)));
      col_strides.push_back(static_cast<size_t>(this->stride(idx)));
    }

    this->rscat = scat(row_lengths, row_strides);
    this->cscat = scat(col_lengths, col_strides);
  }

  std::vector<size_t> rscat;
  std::vector<size_t> cscat;
};

void contract(Tensor<float> A, std::string labelsA,
              Tensor<float> B, std::string labelsB,
              Tensor<float> C, std::string labelsC)
{
  auto indexLabelFinder = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto tensorMatrixA = new TensorMatrix(A, indexLabelFinder->I, indexLabelFinder->Pa);
  auto tensorMatrixB = new TensorMatrix(B, indexLabelFinder->Pb, indexLabelFinder->J);
  auto tensorMatrixC = new TensorMatrix(C, indexLabelFinder->Ic, indexLabelFinder->Jc);

  // B_(PJ)
  auto rscat_I_C = tensorMatrixC->rscat;
  auto cscat_J_C = tensorMatrixC->cscat;

  // A_(IP)
  auto rscat_I_A = tensorMatrixA->rscat;
  auto cscat_P_A = tensorMatrixA->cscat;

  // B_(PJ)
  auto rscat_P_B = tensorMatrixB->rscat;
  auto cscat_J_B = tensorMatrixB->cscat;

  float *A_ = A.data();
  float *B_ = B.data();
  float *C_ = C.data();
  for (int i = 0; i < rscat_I_A.size(); i++)
  {
    for (int j = 0; j < cscat_J_B.size(); j++)
    {
      float c_ij = 0.;
      for (int p = 0; p < cscat_P_A.size(); p++)
      {
        const float a = A_[rscat_I_A[i] + cscat_P_A[p]];
        const float b = B_[rscat_P_B[p] + cscat_J_B[j]];
        c_ij += a * b;
      }

      C_[rscat_I_C[i] + cscat_J_C[j]] = c_ij; // 2 bc stride two for dummy example (TODO: Real address)
    }
  }
}

template <typename T>
void alloc_aligned(T **ptr, size_t n)
{
  // TODO: Memory Alignment?
  if (posix_memalign((void **)ptr, 32, n * sizeof(T)))
  {
    std::throw_with_nested(std::bad_alloc());
  }
}

int main()
{
  /*
  auto A_lengths = {2, 4, 3, 3};
  auto B_lengths = {4, 4, 6};
  auto C_lengths = {6, 3, 2, 3, 4};*/

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

  // C_abcde = A_cfbd . B_fea
  // contract(A, "cfbd", B, "fea", C, "abcde");
  contract(A, "ba", B, "bc", C, "ca");

  std::cout << A << std::endl;
  std::cout << B << std::endl;
  std::cout << C << std::endl;

  return 1;
}