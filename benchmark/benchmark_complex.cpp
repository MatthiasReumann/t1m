#include <iostream>
#include <chrono>
#include <numeric>
#include <complex>
#include "tfctc/tfctc.hpp"

const int N = 10;

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

  std::cout << "d,m,a" << '\n';

  size_t workspace_size = 45l * 45l * 45l * 45l * 45l * 45l;;
  tfctc::utils::alloc_aligned(&A, workspace_size);
  tfctc::utils::alloc_aligned(&B, workspace_size);
  tfctc::utils::alloc_aligned(&C, workspace_size);

  // set_random(A, workspace_size);
  // set_random(B, workspace_size);
  // set_random(C, workspace_size);

  auto a = std::string("abcde");
  auto b = std::string("cijkd");
  auto c = std::string("abeijk");

  for (size_t i = 2; i < 30; i++)
  {
    const std::vector<size_t> lengthsA = { i, i, i, i, i };
    const std::vector<size_t> lengthsB = { i, i, i, i, i };
    const std::vector<size_t> lengthsC = { i, i, i, i, i, i };

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

    std::cout << int(i * i) << ',';
    std::cout << *min_element(time.begin(), time.end()) << ',';
    std::cout << accumulate(time.begin(), time.end(), 0.0) * 1.0 / N << std::endl;
  }

  free(A);
  free(B);
  free(C);
}