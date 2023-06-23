#include <iostream>
#include <numeric>
#include "tfctc/tfctc.hpp"

const int N = 1000;
const int POWER = 11;

template<typename T>
void set_random(T *tensor, size_t size)
{
  for(int i = 0; i < size; i++)
  {
    tensor[i] = static_cast <T> (rand()) / (static_cast <T> (RAND_MAX));
  }
}

int main()
{
  size_t d1, d2, d3;
  float *A = nullptr, *B = nullptr, *C = nullptr;

  std::cout << "d1=d2=d3;min(μs);avg(μs)"<< '\n';
  for (int i = 2; i < POWER; i++)
  {
    d1 = d2 = d3 = pow(2, i);

    tfctc::utils::alloc_aligned(&A, d1 * d2);
    tfctc::utils::alloc_aligned(&B, d2 * d3);
    tfctc::utils::alloc_aligned(&C, d1 * d3);

    set_random(A, d1 * d2);
    set_random(B, d2 * d3);
    set_random(C, d1 * d3);

    const std::vector<size_t> lengthsA = { d1, d2 };
    const std::vector<size_t> lengthsB = { d2, d3 };
    const std::vector<size_t> lengthsC = { d1, d3 };

    auto tensorA = tfctc::Tensor<float>(lengthsA, A);
    auto tensorB = tfctc::Tensor<float>(lengthsB, B);
    auto tensorC = tfctc::Tensor<float>(lengthsC, C);

    std::vector<float> time(N);
    for (uint i = 0; i < N; i++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      
      tfctc::contract(1., tensorA, "ab", tensorB, "cb", 0., tensorC, "ac");
      
      auto t1 = std::chrono::high_resolution_clock::now();
      time[i] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    std::cout << d1 << ';';
    std::cout << *min_element(time.begin(), time.end()) << ';';
    std::cout << accumulate(time.begin(), time.end(), 0.0) * 1.0 / N << std::endl;

    free(A);
    free(B);
    free(C);
  }
}