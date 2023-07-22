#include <iostream>
#include <chrono>
#include <numeric>
#include <string>
#include <complex>
#include<cstdlib>
#include "tfctc/tfctc.hpp"

const int N = 10;
const int ROUNDS = 2000;
const size_t MAX_SIZE = 2000;

class Contraction {
public:
  Contraction(std::string labelC, std::string labelA, std::string labelB,
    std::initializer_list<size_t> lengthsC, std::initializer_list<size_t> lengthsA, std::initializer_list<size_t> lengthsB,
    std::initializer_list<size_t> contractionIndicesA, std::initializer_list<size_t> contractionIndicesB)
    : labelA{ labelA }, labelB{ labelB }, labelC{ labelC },
    lengthsA{ lengthsA }, lengthsB{ lengthsB }, lengthsC{ lengthsC },
    contractionIndicesA{ contractionIndicesA }, contractionIndicesB{ contractionIndicesB } {}

  size_t setRandomContractionLengths()
  {
    size_t lall = 1;
    if (this->contractionIndicesA.size() == 1)
    {
      const auto ai = this->contractionIndicesA.at(0);
      const auto bi = this->contractionIndicesB.at(0);
      const size_t l = 4 + (rand() % MAX_SIZE);
      this->lengthsA[ai] = l;
      this->lengthsB[bi] = l;

      lall *= l;
    }
    else {
      for (int i = 0; i < this->contractionIndicesA.size(); i++) {
        const auto ai = this->contractionIndicesA.at(i);
        const auto bi = this->contractionIndicesB.at(i);
        const size_t l = 4 + (rand() % 45);
        this->lengthsA[ai] = l;
        this->lengthsB[bi] = l;
        lall *= l;
      }
    }
    return lall;
  }

  std::string labelC;
  std::string labelA;
  std::string labelB;
  std::vector<size_t> lengthsC;
  std::vector<size_t> lengthsA;
  std::vector<size_t> lengthsB;

  std::vector<size_t> contractionIndicesA;
  std::vector<size_t> contractionIndicesB;
};

int main()
{
  std::complex<double>* A = nullptr, * B = nullptr, * C = nullptr;
  const std::vector<Contraction> contractions = {
    Contraction("abcde", "efbad", "cf", {24, 16, 12, 16, 24}, {24, 16, 16, 24, 16}, {12, 16}, {1}, {1}),
    Contraction("abcde", "efcad", "bf", {24, 12, 16, 16, 24}, {24, 16, 16, 24, 16}, {12, 16}, {1}, {1}),
    Contraction("abcde", "ecbfa", "fd", {24, 16, 16, 12, 24}, {24, 16, 16, 24, 24}, {24, 12}, {3}, {0}),

    Contraction("abcdef", "dega", "gfbc", {12, 8, 8, 12, 8, 8}, {12, 8, 12, 12}, {12, 8, 8, 8}, {2}, {0}),
    Contraction("abcdef", "dfgb", "geac", {12, 8, 8, 12, 8, 8}, {12, 8, 12, 8}, {12, 8, 12, 8}, {2}, {0}),
    Contraction("abcdef", "degb", "gfac", {12, 8, 8, 12, 8, 8}, {12, 8, 12, 8}, {12, 8, 12, 8}, {2}, {0}),
    Contraction("abcdef", "degc", "gfab", {12, 8, 8, 12, 8, 8}, {12, 8, 12, 8}, {12, 8, 12, 8}, {2}, {0}),

    Contraction("abcd", "ea", "ebcd", {36, 36, 36, 36}, {36, 36}, {36, 36, 36, 36}, {0}, {0}),
    Contraction("abcd", "eb", "aecd", {36, 36, 36, 36}, {36, 36}, {36, 36, 36, 36}, {0}, {1}),
    Contraction("abcd", "ec", "abed", {36, 36, 36, 36}, {36, 36}, {36, 36, 36, 36}, {0}, {2}),

    Contraction("ab", "cad", "dcb", {156, 148}, {156, 156, 156}, {156, 156, 148}, {0, 2}, {1, 0}),
    Contraction("ab", "acd", "dbc", {156, 148}, {156, 148, 156}, {156, 148, 156}, {1, 2}, {0, 2}),
    Contraction("abc", "acd", "db", {156, 156, 148}, {156, 148, 148}, {148, 156}, {2}, {0}),
    Contraction("abc", "adc", "bd", {156, 156, 148}, {156, 148, 148}, {156, 148}, {1}, {1}),
    Contraction("abcd", "aebf", "fdec", {36, 36, 36, 36}, {36, 36, 36, 36}, {36, 36, 36, 36}, {1, 3}, {2, 0}),
    Contraction("abcd", "eafd", "fbec", {36, 36, 36, 36}, {36, 36, 36, 36}, {36, 36, 36, 36}, {0, 2}, {2, 0}),
    // Contraction("abcd", "aebf", "dfce", {36, 36, 36, 36}, {36, 36, 36, 36}, {36, 36, 36, 36}, {1, 3}, {3, 1})

  };

  std::cout << "size,contraction,min" << std::endl;
  for (int r = 0; r < ROUNDS; r++)
  {
    const size_t i = rand() % contractions.size();

    auto contraction = contractions.at(i);
    auto contraction_size = contraction.setRandomContractionLengths();

    size_t sizeA = 1L;
    size_t sizeB = 1L;
    size_t sizeC = 1L;

    auto lengthsC = contraction.lengthsC;
    auto lengthsA = contraction.lengthsA;
    auto lengthsB = contraction.lengthsB;

    for (auto& l : lengthsA) sizeA *= l;
    for (auto& l : lengthsB) sizeB *= l;
    for (auto& l : lengthsC) sizeC *= l;

    tfctc::utils::alloc_aligned(&A, sizeA);
    tfctc::utils::alloc_aligned(&B, sizeB);
    tfctc::utils::alloc_aligned(&C, sizeC);

    auto tensorA = tfctc::Tensor<std::complex<double>>(lengthsA, A);
    auto tensorB = tfctc::Tensor<std::complex<double>>(lengthsB, B);
    auto tensorC = tfctc::Tensor<std::complex<double>>(lengthsC, C);

    std::cout << contraction_size << ',' << contraction.labelC << '-' << contraction.labelA << '-' << contraction.labelB << ',';
    std::vector<double> time(N);
    for (uint j = 0; j < N; j++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      tfctc::contract(tensorA, contraction.labelA.c_str(), tensorB, contraction.labelB.c_str(), tensorC, contraction.labelC.c_str());
      auto t1 = std::chrono::high_resolution_clock::now();
      time[j] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    std::cout << *min_element(time.begin(), time.end()) << std::endl;

    free(A);
    free(B);
    free(C);
  }
}