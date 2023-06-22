#include <iostream>
#include <numeric>
#include <complex>
#include "tfctc.hpp"

const int N = 16;
const int POWER = 12;

template<typename T>
void set_random(std::complex<T> *tensor, size_t size)
{
  for(int i = 0; i < size; i++)
  {
    tensor[i].real(static_cast <T> (rand()) / (static_cast <T> (RAND_MAX)));
    tensor[i].imag(static_cast <T> (rand()) / (static_cast <T> (RAND_MAX)));
  }
}

int main()
{
  int d1, d2, d3;
  std::complex<float>* A = nullptr, * B = nullptr, * C = nullptr;
  
  std::cout << "d1=d2=d3;min(μs);avg(μs)"<< '\n';
  for (int i = 10; i < 11; i++)
  {
    d1 = d2 = d3 = pow(2, i);

    tfctc::utils::alloc_aligned(&A, d1 * d2);
    tfctc::utils::alloc_aligned(&B, d2 * d3);
    tfctc::utils::alloc_aligned(&C, d1 * d3);

    set_random(A, d1 * d2);
    set_random(B, d2 * d3);
    set_random(C, d1 * d3);

    auto tensorA = tfctc::Tensor<std::complex<float>>({ d1, d2 }, A);
    auto tensorB = tfctc::Tensor<std::complex<float>>({ d2, d3 }, B);
    auto tensorC = tfctc::Tensor<std::complex<float>>({ d1, d3 }, C);

    std::vector<float> time(N);
    for (uint i = 0; i < N; i++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      tfctc::contract(tensorA, "ab", tensorB, "cb", tensorC, "ac");
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