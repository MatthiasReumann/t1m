#include <iostream>
#include "tfctc.hpp"

constexpr int KP = 4;
constexpr int MR = 6;
constexpr int NR = 8;
constexpr int MC = 72;
constexpr int NC = 4080;
constexpr int KC = 256;

constexpr char NWLN = '\n';

void test4x4()
{
  std::cout << "TEST 4x4 . 4x4 = 4x4" << '\n';
  float *A_ptr = nullptr;
  float *B_ptr = nullptr;
  float *C_ptr = nullptr;

  alloc_aligned<float>(&A_ptr, 4 * 4);
  alloc_aligned<float>(&B_ptr, 4 * 4);
  alloc_aligned<float>(&C_ptr, 4 * 4);

  A_ptr[0] = 3.;
  A_ptr[4] = 1.;
  A_ptr[8] = 0.;
  A_ptr[12] = 1.;
  A_ptr[1] = 4.;
  A_ptr[5] = 7.;
  A_ptr[9] = 1.1;
  A_ptr[13] = 1.;
  A_ptr[2] = 1.;
  A_ptr[6] = 0.;
  A_ptr[10] = 1.;
  A_ptr[14] = 1.;
  A_ptr[3] = 1.;
  A_ptr[7] = 0.;
  A_ptr[11] = 0.;
  A_ptr[15] = 1.;

  B_ptr[0] = 1.;
  B_ptr[4] = 1.;
  B_ptr[8] = 0.;
  B_ptr[12] = 1.;
  B_ptr[1] = 0.;
  B_ptr[5] = 0.;
  B_ptr[9] = 1.;
  B_ptr[13] = 1.7;
  B_ptr[2] = 1.;
  B_ptr[6] = 1.;
  B_ptr[10] = 4.3;
  B_ptr[14] = 2.;
  B_ptr[3] = 1.;
  B_ptr[7] = 3.;
  B_ptr[11] = 1.;
  B_ptr[15] = 1.;

  auto A_lengths = {4, 4};
  auto B_lengths = {4, 4};
  auto C_lengths = {4, 4};

  auto A = Tensor<float>(A_lengths, A_ptr);
  auto B = Tensor<float>(B_lengths, B_ptr);
  auto C = Tensor<float>(C_lengths, C_ptr);

  std::cout << A << std::endl;
  std::cout << B << std::endl;

  contract(1., A, "ab", B, "bc", 0., C, "ca");

  std::cout << C << std::endl;
}

void test4x3()
{
  std::cout << "TEST 3x4 . 4x3 = 3x3" << '\n';
  float *A_ptr = nullptr;
  float *B_ptr = nullptr;
  float *C_ptr = nullptr;

  alloc_aligned<float>(&A_ptr, 4 * 3);
  alloc_aligned<float>(&B_ptr, 4 * 3);
  alloc_aligned<float>(&C_ptr, 3 * 3);

  A_ptr[0] = 3.;
  A_ptr[4] = 1.;
  A_ptr[8] = 0.;
  A_ptr[1] = 4.;
  A_ptr[5] = 7.;
  A_ptr[9] = 1.1;
  A_ptr[2] = 1.;
  A_ptr[6] = 0.;
  A_ptr[10] = 1.;
  A_ptr[3] = 1.;
  A_ptr[7] = 0.;
  A_ptr[11] = 0.;

  B_ptr[0] = 1.;
  B_ptr[4] = 1.;
  B_ptr[8] = 0.;
  B_ptr[2] = 0.;
  B_ptr[5] = 0.;
  B_ptr[9] = 1.;
  B_ptr[1] = 1.;
  B_ptr[6] = 1.;
  B_ptr[10] = 4.3;
  B_ptr[3] = 1.;
  B_ptr[7] = 3.;
  B_ptr[11] = 1.;

  auto A_lengths = {4, 3};
  auto B_lengths = {4, 3};
  auto C_lengths = {3, 3};

  auto A = Tensor<float>(A_lengths, A_ptr);
  auto B = Tensor<float>(B_lengths, B_ptr);
  auto C = Tensor<float>(C_lengths, C_ptr);

  std::cout << A << std::endl;
  std::cout << B << std::endl;

  contract(1., A, "ba", B, "bc", 0., C, "ac");

  std::cout << C << std::endl;
}

void testpaper()
{
  auto A_lengths = {2, 4, 3, 3};
  auto B_lengths = {4, 4, 6};
  auto C_lengths = {6, 3, 2, 3, 4};

  float *A_ptr = nullptr;
  float *B_ptr = nullptr;
  float *C_ptr = nullptr;

  alloc_aligned<float>(&A_ptr, 2 * 4 * 3 * 3);
  alloc_aligned<float>(&B_ptr, 4 * 4 * 6);
  alloc_aligned<float>(&C_ptr, 6 * 3 * 2 * 3 * 4);

  std::cout << "sizeof(float) = " << sizeof(float) << std::endl;

  auto A = Tensor<float>(A_lengths, A_ptr);
  auto B = Tensor<float>(B_lengths, B_ptr);
  auto C = Tensor<float>(C_lengths, C_ptr);

  // C_abcde = A_cfbd . B_fea
  contract(1., A, "cfbd", B, "fea", 0., C, "abcde");

  /*
  std::cout << A << std::endl;
  std::cout << B << std::endl;
  std::cout << C << std::endl;*/
}

void test_macrokernel_simple()
{
  float *A, *B, *C;

  alloc_aligned<float>(&A, 2 * 2);
  alloc_aligned<float>(&B, 2 * 2);
  alloc_aligned<float>(&C, 2 * 2);

  A[0] = 1.;
  A[2] = 2.7;
  A[1] = 3.;
  A[3] = 1.0;
  B[0] = 1.;
  B[2] = 1.;
  B[1] = 2.;
  B[3] = 0.;

  macrokernel_simple<2, 2, 2>(A, B, C);

  print_mat(C, 2, 2);
}

void benchmark()
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

int main()
{
  // test4x4();
  // test4x3();
  // testpaper();
  benchmark();

  return 1;
}