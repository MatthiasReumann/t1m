#include <iostream>
#include <chrono>
#include <numeric>
#include <string>
#include <complex>
#include <cstdlib>
#include "t1m/t1m.hpp"

constexpr int N = 10;
constexpr size_t MAX_SIZE = 1000;

class Contraction {
public:
  Contraction(const std::string &labelC, const std::string &labelA, const std::string &labelB,
    std::initializer_list<size_t> lengthsC, std::initializer_list<size_t> lengthsA, std::initializer_list<size_t> lengthsB,
    std::initializer_list<size_t> contractionIndicesA, std::initializer_list<size_t> contractionIndicesB)
    : labelA{ labelA }, labelB{ labelB }, labelC{ labelC },
    lengthsA{ lengthsA }, lengthsB{ lengthsB }, lengthsC{ lengthsC },
    contractionIndicesA{ contractionIndicesA }, contractionIndicesB{ contractionIndicesB } {}

  size_t set_contraction_size(size_t s)
  {
    size_t lall = 1;
    for (int i = 0; i < this->contractionIndicesA.size(); i++) {
      const auto ai = this->contractionIndicesA.at(i);
      const auto bi = this->contractionIndicesB.at(i);
      this->lengthsA[ai] = s;
      this->lengthsB[bi] = s;
      lall *= s;
    }
    return lall;
  }

  std::string labelA;
  std::string labelB;
  std::string labelC;
  std::vector<size_t> lengthsA;
  std::vector<size_t> lengthsB;
  std::vector<size_t> lengthsC;

  std::vector<size_t> contractionIndicesA;
  std::vector<size_t> contractionIndicesB;
};

void run(Contraction contraction)
{
  size_t contraction_size, sizeA, sizeB, sizeC;
  std::complex<double>* A = nullptr, * B = nullptr, * C = nullptr;

  std::cout << "size,contraction,min" << std::endl;
  for(size_t i = 0; i < MAX_SIZE; i += 20)
  {
    contraction_size = contraction.set_contraction_size(i);

    sizeA = 1L;
    sizeB = 1L;
    sizeC = 1L;

    const auto &lengthsC = contraction.lengthsC;
    const auto &lengthsA = contraction.lengthsA;
    const auto &lengthsB = contraction.lengthsB;

    for (const auto& l : lengthsA) sizeA *= l;
    for (const auto& l : lengthsB) sizeB *= l;
    for (const auto& l : lengthsC) sizeC *= l;

    t1m::utils::alloc_aligned(&A, sizeA);
    t1m::utils::alloc_aligned(&B, sizeB);
    t1m::utils::alloc_aligned(&C, sizeC);

    auto tensorA = t1m::Tensor<std::complex<double>>(lengthsA, A);
    auto tensorB = t1m::Tensor<std::complex<double>>(lengthsB, B);
    auto tensorC = t1m::Tensor<std::complex<double>>(lengthsC, C);

    std::cout << contraction_size << ',' << contraction.labelC << '-' << contraction.labelA << '-' << contraction.labelB << ',';
    std::vector<double> time(N);
    for (size_t j = 0; j < N; j++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      t1m::contract(tensorA, contraction.labelA.c_str(), tensorB, contraction.labelB.c_str(), tensorC, contraction.labelC.c_str());
      auto t1 = std::chrono::high_resolution_clock::now();
      time[j] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    std::cout << *min_element(time.begin(), time.end()) << std::endl;

    free(A);
    free(B);
    free(C);
  }
}

void worst()
{
  run(Contraction("abcdef", "dega", "gfbc", {12, 8, 8, 12, 8, 8}, {12, 8, 12, 12}, {12, 8, 8, 8}, {2}, {0}));
}

void best()
{
  run(Contraction("abc", "bda", "dc", {156, 156, 12}, {156, 156, 156}, {156, 12}, {1}, {0}));
}

void gemm()
{
  run(Contraction("ab", "ac", "cb", {2568, 2560}, {2568, 2568}, {2568, 2560}, {1}, {0}));
}

void notrans()
{
  run(Contraction("abcd", "ea", "ebcd", {128,64,64,64}, {64,128}, {64, 64, 64, 64}, {0}, {0}));
}

int main()
{
  // notrans();
  worst();
  best();
  gemm();
}