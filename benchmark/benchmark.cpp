#include <iostream>
#include "tfctc.hpp"

int main()
{
  const int N = 1000;
  float *A = nullptr, *B = nullptr, *C = nullptr;

  for (int i = 0; i < 11; i++)
  {
    int d1 = pow(2, i), d2 = pow(2, i), d3 = pow(2, i);

    alloc_aligned(&A, d1 * d2);
    alloc_aligned(&B, d2 * d3);
    alloc_aligned(&C, d1 * d3);

    auto tensorA = Tensor<float>({d1, d2}, A);
    auto tensorB = Tensor<float>({d2, d3}, B);
    auto tensorC = Tensor<float>({d1, d3}, C);

    std::vector<float> time(N);
    for (uint i = 0; i < N; i++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      contract(1., tensorA, "ab", tensorB, "cb", 0., tensorC, "ac");
      auto t1 = std::chrono::high_resolution_clock::now();
      time[i] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    std::cout << "TEST: d1=" << d1 << " d2=" << d2 << " d3=" << d3 << std::endl;
    std::cout << "Minimum: " << *min_element(time.begin(), time.end()) << " ms" << std::endl;
    std::cout << "Average: " << accumulate(time.begin(), time.end(), 0.0) * 1.0 / N << " ms" << std::endl;

    free(A);
    free(B);
    free(C);
  }
}