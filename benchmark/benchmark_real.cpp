#include <iostream>
#include <chrono>
#include <numeric>
#include "tfctc/tfctc.hpp"

const int N = 10;

template<typename T>
void set_random(T* tensor, size_t size)
{
  for (int i = 0; i < size; i++)
  {
    tensor[i] = static_cast <T> (rand()) / (static_cast <T> (RAND_MAX));
  }
}

void test(float* A, std::string labelsA, float* B, std::string labelsB, float* C, std::string labelsC)
{
  for (size_t i = 2; i < 45; i++)
  {
    const std::vector<size_t> lengthsA = { i, i, i, i};
    const std::vector<size_t> lengthsB = { i, i, i, i};
    const std::vector<size_t> lengthsC = { i, i, i, i};

    auto tensorA = tfctc::Tensor<float>(lengthsA, A);
    auto tensorB = tfctc::Tensor<float>(lengthsB, B);
    auto tensorC = tfctc::Tensor<float>(lengthsC, C);

    std::vector<float> time(N);
    for (uint j = 0; j < N; j++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      tfctc::contract(1., tensorA, labelsA, tensorB, labelsB, 0., tensorC, labelsC);
      auto t1 = std::chrono::high_resolution_clock::now();
      time[j] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    std::cout << int(i * i) << ',';
    std::cout << *min_element(time.begin(), time.end()) << ',';
    std::cout << accumulate(time.begin(), time.end(), 0.0) * 1.0 / N << std::endl;
  }
}

int main()
{
  float* A = nullptr, * B = nullptr, * C = nullptr;

  std::cout << "d,m,a" << '\n';

  size_t workspace_size = 45l * 45l * 45l * 45l;
  tfctc::utils::alloc_aligned(&A, workspace_size);
  tfctc::utils::alloc_aligned(&B, workspace_size);
  tfctc::utils::alloc_aligned(&C, workspace_size);

  // set_random(A, workspace_size);
  // set_random(B, workspace_size);

  auto a = std::string("abcd");
  auto b = std::string("ebcf");
  auto c = std::string("adef");

  test(A, a, B, b, C, c);
}