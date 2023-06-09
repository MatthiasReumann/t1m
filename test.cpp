#include <iostream>
#include <complex>
#include "tfctc.hpp"

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

  contract(1., A, "ba", B, "bc", 0., C, "ac");

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


void test2x2() {
  std::cout << "TEST 2x2 . 2x2 = 2x2" << '\n';
  float *A_ptr = nullptr;
  float *B_ptr = nullptr;
  float *C_ptr = nullptr;

  alloc_aligned<float>(&A_ptr, 2 * 2);
  alloc_aligned<float>(&B_ptr, 2 * 2);
  alloc_aligned<float>(&C_ptr, 2 * 2);

  A_ptr[0] = 3.; A_ptr[2] = 0.;
  A_ptr[1] = 1.; A_ptr[3] = 1.7;

  B_ptr[0] = 1.; B_ptr[2] = 3.4;
  B_ptr[1] = 1.; B_ptr[3] = 1;

  auto A_lengths = {2, 2};
  auto B_lengths = {2, 2};
  auto C_lengths = {2, 2};

  auto A = Tensor<float>(A_lengths, A_ptr);
  auto B = Tensor<float>(B_lengths, B_ptr);
  auto C = Tensor<float>(C_lengths, C_ptr);

  std::cout << A << std::endl;
  std::cout << B << std::endl;

  contract(1., A, "ab", B, "cb", 0., C, "ac");

  std::cout << C << std::endl;
}


void test2x2_complex() {
  std::cout << "TEST (complex) 2x2 . 2x2 = 2x2" << '\n';
  std::complex<float> *A_ptr = nullptr;
  std::complex<float> *B_ptr = nullptr;
  std::complex<float> *C_ptr = nullptr;

  alloc_aligned<std::complex<float>>(&A_ptr, 2 * 2);
  alloc_aligned<std::complex<float>>(&B_ptr, 2 * 2);
  alloc_aligned<std::complex<float>>(&C_ptr, 2 * 2);

  A_ptr[0] = std::complex(1.); A_ptr[2] = std::complex(3., 1.);
  A_ptr[1] = std::complex(2.); A_ptr[3] = std::complex(0.);

  B_ptr[0] = std::complex(1.); B_ptr[2] = std::complex(0.);
  B_ptr[1] = std::complex(0.); B_ptr[3] = std::complex(1.);

  auto A_lengths = {2, 2};
  auto B_lengths = {2, 2};
  auto C_lengths = {2, 2};

  auto A = Tensor<std::complex<float>>(A_lengths, A_ptr);
  auto B = Tensor<std::complex<float>>(B_lengths, B_ptr);
  auto C = Tensor<std::complex<float>>(C_lengths, C_ptr);

  std::cout << A << std::endl;
  std::cout << B << std::endl;

  contract(A, "ab", B, "bc", C, "ac");

  std::cout << C << std::endl;
}

void test512x512_complex() {
  std::cout << "TEST (complex) 512x512 . 512x512 = 512x512" << '\n';
  std::complex<float> *A_ptr = nullptr;
  std::complex<float> *B_ptr = nullptr;
  std::complex<float> *C_ptr = nullptr;

  alloc_aligned<std::complex<float>>(&A_ptr, 512 * 512);
  alloc_aligned<std::complex<float>>(&B_ptr, 512 * 512);
  alloc_aligned<std::complex<float>>(&C_ptr, 512 * 512);

  A_ptr[0] = std::complex(1., 1.); A_ptr[4] = std::complex(5., 5.); A_ptr[8] = std::complex(8.); A_ptr[12] = std::complex(11.);
  A_ptr[1] = std::complex(2., 2.); A_ptr[5] = std::complex(1., 1.); A_ptr[9] = std::complex(9.); A_ptr[13] = std::complex(12.);
  A_ptr[2] = std::complex(3., 3.); A_ptr[6] = std::complex(6., 6.); A_ptr[10] = std::complex(1.); A_ptr[14] = std::complex(13.);
  A_ptr[3] = std::complex(4., 4.); A_ptr[7] = std::complex(7., 7.); A_ptr[11] = std::complex(10.); A_ptr[15] = std::complex(0);

  B_ptr[0] = std::complex(1.,0.); B_ptr[4] = std::complex(0.); B_ptr[8] = std::complex(0.); B_ptr[12] = std::complex(0.);
  B_ptr[1] = std::complex(0.); B_ptr[5] = std::complex(1.,0.); B_ptr[9] = std::complex(0.); B_ptr[13] = std::complex(0.);
  B_ptr[2] = std::complex(0.); B_ptr[6] = std::complex(0.); B_ptr[10] = std::complex(1.,0.); B_ptr[14] = std::complex(0.);
  B_ptr[3] = std::complex(0.); B_ptr[7] = std::complex(0.); B_ptr[11] = std::complex(0.); B_ptr[15] = std::complex(1.,0.);

  auto A_lengths = {512, 512};
  auto B_lengths = {512, 512};
  auto C_lengths = {512, 512};
  auto A = Tensor<std::complex<float>>(A_lengths, A_ptr);
  auto B = Tensor<std::complex<float>>(B_lengths, B_ptr);
  auto C = Tensor<std::complex<float>>(C_lengths, C_ptr);

  // std::cout << A << std::endl;
  // std::cout << B << std::endl;

  contract(A, "ab", B, "bc", C, "ca");

  // std::cout << C << std::endl;
}

void test1024x1024_complex() {
  std::cout << "TEST (complex) 1024x1024 . 1024x1024 = 1024x1024" << '\n';
  std::complex<float> *A_ptr = nullptr;
  std::complex<float> *B_ptr = nullptr;
  std::complex<float> *C_ptr = nullptr;

  alloc_aligned<std::complex<float>>(&A_ptr, 1024 * 1024);
  alloc_aligned<std::complex<float>>(&B_ptr, 1024 * 1024);
  alloc_aligned<std::complex<float>>(&C_ptr, 1024 * 1024);

  A_ptr[0] = std::complex(1., 1.); A_ptr[4] = std::complex(5., 5.); A_ptr[8] = std::complex(8.); A_ptr[12] = std::complex(11.);
  A_ptr[1] = std::complex(2., 2.); A_ptr[5] = std::complex(1., 1.); A_ptr[9] = std::complex(9.); A_ptr[13] = std::complex(12.);
  A_ptr[2] = std::complex(3., 3.); A_ptr[6] = std::complex(6., 6.); A_ptr[10] = std::complex(1.); A_ptr[14] = std::complex(13.);
  A_ptr[3] = std::complex(4., 4.); A_ptr[7] = std::complex(7., 7.); A_ptr[11] = std::complex(10.); A_ptr[15] = std::complex(0);

  B_ptr[0] = std::complex(1.,0.); B_ptr[4] = std::complex(0.); B_ptr[8] = std::complex(0.); B_ptr[12] = std::complex(0.);
  B_ptr[1] = std::complex(0.); B_ptr[5] = std::complex(1.,0.); B_ptr[9] = std::complex(0.); B_ptr[13] = std::complex(0.);
  B_ptr[2] = std::complex(0.); B_ptr[6] = std::complex(0.); B_ptr[10] = std::complex(1.,0.); B_ptr[14] = std::complex(0.);
  B_ptr[3] = std::complex(0.); B_ptr[7] = std::complex(0.); B_ptr[11] = std::complex(0.); B_ptr[15] = std::complex(1.,0.);

  auto A_lengths = {1024, 1024};
  auto B_lengths = {1024, 1024};
  auto C_lengths = {1024, 1024};
  auto A = Tensor<std::complex<float>>(A_lengths, A_ptr);
  auto B = Tensor<std::complex<float>>(B_lengths, B_ptr);
  auto C = Tensor<std::complex<float>>(C_lengths, C_ptr);

  // std::cout << A << std::endl;
  // std::cout << B << std::endl;

  contract(A, "ab", B, "bc", C, "ca");

  // std::cout << C << std::endl;
}

void test4x4_complex() {
  std::cout << "TEST (complex) 4x4 . 4x4 = 4x4" << '\n';
  std::complex<float> *A_ptr = nullptr;
  std::complex<float> *B_ptr = nullptr;
  std::complex<float> *C_ptr = nullptr;

  alloc_aligned<std::complex<float>>(&A_ptr, 4 * 4);
  alloc_aligned<std::complex<float>>(&B_ptr, 4 * 4);
  alloc_aligned<std::complex<float>>(&C_ptr, 4 * 4);

  A_ptr[0] = std::complex(1., 1.); A_ptr[4] = std::complex(5., 5.); A_ptr[8] = std::complex(8.); A_ptr[12] = std::complex(11.);
  A_ptr[1] = std::complex(2., 2.); A_ptr[5] = std::complex(1., 1.); A_ptr[9] = std::complex(9.); A_ptr[13] = std::complex(12.);
  A_ptr[2] = std::complex(3., 3.); A_ptr[6] = std::complex(6., 6.); A_ptr[10] = std::complex(1.); A_ptr[14] = std::complex(13.);
  A_ptr[3] = std::complex(4., 4.); A_ptr[7] = std::complex(7., 7.); A_ptr[11] = std::complex(10.); A_ptr[15] = std::complex(0);

  B_ptr[0] = std::complex(1.,0.); B_ptr[4] = std::complex(0.); B_ptr[8] = std::complex(0.); B_ptr[12] = std::complex(0.);
  B_ptr[1] = std::complex(0.); B_ptr[5] = std::complex(1.,0.); B_ptr[9] = std::complex(0.); B_ptr[13] = std::complex(0.);
  B_ptr[2] = std::complex(0.); B_ptr[6] = std::complex(0.); B_ptr[10] = std::complex(1.,0.); B_ptr[14] = std::complex(0.);
  B_ptr[3] = std::complex(0.); B_ptr[7] = std::complex(0.); B_ptr[11] = std::complex(0.); B_ptr[15] = std::complex(1.,0.);

  auto A_lengths = {4, 4};
  auto B_lengths = {4, 4};
  auto C_lengths = {4, 4};
  auto A = Tensor<std::complex<float>>(A_lengths, A_ptr);
  auto B = Tensor<std::complex<float>>(B_lengths, B_ptr);
  auto C = Tensor<std::complex<float>>(C_lengths, C_ptr);

  std::cout << A << std::endl;
  std::cout << B << std::endl;

  contract(A, "ab", B, "bc", C, "ca");

  std::cout << C << std::endl;
}

int main()
{
  test2x2();
  test4x4();
  test4x3();
  test2x2_complex();
  test4x4_complex();
  test512x512_complex();
  test1024x1024_complex();
  // testpaper();

  return 1;
}