#include <iostream>
#include <chrono>
#include <numeric>
#include <complex>
#include <tuple>
#include "tfctc/tfctc.hpp"

const int N = 10;
const size_t L = 6L;

template<typename T>
void set_random(std::complex<T>* tensor, size_t size)
{
  for (int i = 0; i < size; i++)
  {
    tensor[i].real(static_cast <T> (rand()) / (static_cast <T> (RAND_MAX)));
    tensor[i].imag(static_cast <T> (rand()) / (static_cast <T> (RAND_MAX)));
  }
}

int main()
{
  std::complex<float>* A = nullptr, * B = nullptr, * C = nullptr;
  size_t workspace_size = L * L * L * L * L * L * L * L * L * L * L * L * L * L;

  const std::vector<std::tuple<std::string, std::string, std::string>> contractions = {
        {"abcde", "efbad", "cf"},
        {"abcde", "efcad", "bf"},
        {"abcd", "dbea", "ec"},
        {"abcde", "ecbfa", "fd"},
        {"abcd", "deca", "be"},
        {"abc", "bda", "dc"},
        {"abcd", "ebad", "ce"},
        {"abcdef", "dega", "gfbc"},
        {"abcdef", "dfgb", "geac"},
        {"abcdef", "degb", "gfac"},
        {"abcdef", "degc", "gfab"},
        {"abc", "dca", "bd"},
        {"abcd", "ea", "ebcd"},
        {"abcd", "eb", "aecd"},
        {"abcd", "ec", "abed"},
        {"abc", "adec", "ebd"},
        {"ab", "cad", "dcb"},
        {"ab", "acd", "dbc"},
        {"abc", "acd", "db"},
        {"abc", "adc", "bd"},
        {"ab", "ac", "cb"},
        {"abcd", "aebf", "fdec"},
        {"abcd", "eafd", "fbec"},
        {"abcd", "aebf", "dfce"}
  };

  tfctc::utils::alloc_aligned(&A, workspace_size);
  tfctc::utils::alloc_aligned(&B, workspace_size);
  tfctc::utils::alloc_aligned(&C, workspace_size);

  std::cout << "contraction,min" << '\n';
  for (auto [a, b, c] : contractions)
  {
    std::cout << a << "-" << b << "-" << c << ",";
    
    std::vector<size_t> lengthsA;
    std::vector<size_t> lengthsB;
    std::vector<size_t> lengthsC;

    for (size_t j = 0; j < a.length(); j++)
      lengthsA.push_back(L);

    for (size_t j = 0; j < b.length(); j++)
      lengthsB.push_back(L);

    for (size_t j = 0; j < c.length(); j++)
      lengthsC.push_back(L);

    auto tensorA = tfctc::Tensor<std::complex<float>>(lengthsA, A);
    auto tensorB = tfctc::Tensor<std::complex<float>>(lengthsB, B);
    auto tensorC = tfctc::Tensor<std::complex<float>>(lengthsC, C);

    std::vector<float> time(N);
    for (uint j = 0; j < N; j++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      tfctc::contract(tensorA, a, tensorB, b, tensorC, c);
      auto t1 = std::chrono::high_resolution_clock::now();
      time[j] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    std::cout << *min_element(time.begin(), time.end()) << '\n';
  }

  free(A);
  free(B);
  free(C);
}