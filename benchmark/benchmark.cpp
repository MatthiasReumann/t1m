#include <iostream>
#include <complex>
#include "tfctc.hpp"

void benchmark() {
  const int N = 1000;
  float *A = nullptr, *B = nullptr, *C = nullptr;

  for (int i = 0; i < 11; i++)
  {
    int d1 = pow(2, i), d2 = pow(2, i), d3 = pow(2, i);

    tfctc::utils::alloc_aligned(&A, d1 * d2);
    tfctc::utils::alloc_aligned(&B, d2 * d3);
    tfctc::utils::alloc_aligned(&C, d1 * d3);

    memset(A, 1, d1 * d2 * sizeof(float));
    memset(B, 1, d2 * d3 * sizeof(float));
    memset(C, 1, d1 * d3 * sizeof(float));

    auto tensorA = tfctc::Tensor<float>({d1, d2}, A);
    auto tensorB = tfctc::Tensor<float>({d2, d3}, B);
    auto tensorC = tfctc::Tensor<float>({d1, d3}, C);

    std::vector<float> time(N);
    for (uint i = 0; i < N; i++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      tfctc::contract(1., tensorA, "ab", tensorB, "cb", 0., tensorC, "ac");
      auto t1 = std::chrono::high_resolution_clock::now();
      time[i] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    std::cout << "TEST: d1=" << d1 << " d2=" << d2 << " d3=" << d3 << std::endl;
    std::cout << "Minimum: " << *min_element(time.begin(), time.end()) << " μs" << std::endl;
    std::cout << "Average: " << accumulate(time.begin(), time.end(), 0.0) * 1.0 / N << " μs" << std::endl;

    free(A);
    free(B);
    free(C);
  }
}

void benchmark_complex() {
  const int N = 16;
  std::complex<float> *A = nullptr, *B = nullptr, *C = nullptr;

  for (int i = 0; i < 12; i++)
  {
    int d1 = pow(2, i), d2 = pow(2, i), d3 = pow(2, i);

    tfctc::utils::alloc_aligned(&A, d1 * d2);
    tfctc::utils::alloc_aligned(&B, d2 * d3);
    tfctc::utils::alloc_aligned(&C, d1 * d3);

    memset(A, 1, d1 * d2 * sizeof(std::complex<float>));
    memset(B, 1, d2 * d3 * sizeof(std::complex<float>));
    memset(C, 1, d1 * d3 * sizeof(std::complex<float>));

    auto tensorA = tfctc::Tensor<std::complex<float>>({d1, d2}, A);
    auto tensorB = tfctc::Tensor<std::complex<float>>({d2, d3}, B);
    auto tensorC = tfctc::Tensor<std::complex<float>>({d1, d3}, C);

    std::vector<float> time(N);
    for (uint i = 0; i < N; i++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      tfctc::contract(tensorA, "ab", tensorB, "cb", tensorC, "ac");
      auto t1 = std::chrono::high_resolution_clock::now();
      time[i] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    std::cout << "TEST: d1=" << d1 << " d2=" << d2 << " d3=" << d3 << std::endl;
    std::cout << "Minimum: " << *min_element(time.begin(), time.end()) << " μs" << std::endl;
    std::cout << "Average: " << accumulate(time.begin(), time.end(), 0.0) * 1.0 / N << " μs" << std::endl;

    free(A);
    free(B);
    free(C);
  }
}

int main()
{
  // benchmark();
  benchmark_complex();
}