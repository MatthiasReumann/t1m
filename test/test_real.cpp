#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tfctc.hpp"

template<typename T>
inline void requireAll(T *tensor, std::vector<T> expected)
{
  for(int i = 0; i < expected.size(); i++)
  {
    REQUIRE(tensor[i] == doctest::Approx(expected[i]).epsilon(0.001));
  }
}

TEST_CASE("(float) 2D . ID")
{
  float *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = 3.; A[2] = 4.;
  A[1] = 0.; A[3] = 1.;

  B[0] = 1.; B[2] = 0.;
  B[1] = 0.; B[3] = 1.;

  auto tensorA = tfctc::Tensor<float>({2, 2}, A);
  auto tensorB = tfctc::Tensor<float>({2, 2}, B);
  auto tensorC = tfctc::Tensor<float>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(1.0, tensorA, "ab", tensorB, "bc", 0., tensorC, "ac");
    requireAll(C, {
      3.,
      0.,
      4.,
      1.
    });
  }

  memset(C, 0, 2 * 2 * sizeof(float));

  SUBCASE("transposed")
  {
    tfctc::contract(1.0, tensorA, "ab", tensorB, "bc", 0., tensorC, "ca");
    requireAll(C, {
      3.,
      4.,
      0.,
      1.
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("(float) 2D . 2D => 2D")
{
  float *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = 3.; A[2] = 4.;
  A[1] = 1.; A[3] = 0.;

  B[0] = 1.; B[2] = 0.7;
  B[1] = 0.3; B[3] = 1.;

  auto tensorA = tfctc::Tensor<float>({2, 2}, A);
  auto tensorB = tfctc::Tensor<float>({2, 2}, B);
  auto tensorC = tfctc::Tensor<float>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(1., tensorA, "ab", tensorB, "bc", 0., tensorC, "ac");

    requireAll(C, {
      4.2,
      1.,
      6.1,
      0.7
    });
  }

  memset(C, 0, 2 * 2 * sizeof(float));

  SUBCASE("label order 2")
  {
    tfctc::contract(1., tensorA, "ab", tensorB, "cb", 0., tensorC, "ac");
    requireAll(C, {
      5.8,
      1.,
      4.9,
      0.3,
    });
  }

  memset(C, 0, 2 * 2 * sizeof(float));

  SUBCASE("label order 3")
  {
    tfctc::contract(1., tensorA, "ba", tensorB, "cb", 0., tensorC, "ac");
    requireAll(C, {
      3.7,
      4,
      1.9,
      1.2,
    });
  }

  memset(C, 0, 2 * 2 * sizeof(float));

  SUBCASE("label order 4")
  {
    tfctc::contract(1., tensorA, "ba", tensorB, "bc", 0., tensorC, "ac");
    requireAll(C, {
      3.3,
      4,
      3.1,
      2.8
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("float) 3D . 3D => 2D")
{
  float *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = 3.; A[2] = 3.;
  A[1] = 0;  A[3] = 1.7;
  
  A[4] = -1; A[6] = 1.7;
  A[5] = 3.; A[7] = 0.;


  B[0] = 0.33; B[2] = 0.;
  B[1] = 3;    B[3] = 4;

  B[4] = 1.;   B[6] = 4.;
  B[5] = 0.; B[7] = 0.;

  auto tensorA = tfctc::Tensor<float>({2, 2, 2}, A);
  auto tensorB = tfctc::Tensor<float>({2, 2, 2}, B);
  auto tensorC = tfctc::Tensor<float>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(1.0, tensorA, "abc", tensorB, "cbd", 0., tensorC, "ad");
    requireAll(C, {
      4.79,
      9,
      15,
      6.8
    });
  }

  memset(C, 0, 2 * 2 * sizeof(float));

  SUBCASE("transposed")
  {
    tfctc::contract(1., tensorA, "abc", tensorB, "cbd", 0., tensorC, "da");
    requireAll(C, {
      4.79,
      15,
      9,
      6.8
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("(float) 3D . 2D => 3D")
{
  float *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2 * 2);

  A[0] = 0.1337; A[2] = 0.;
  A[1] = 3.; A[3] = 1.7;

  A[4] = 5.3; A[6] = 7.5;
  A[5] = 3.3; A[7] = 1.33;

  B[0] = 1.; B[2] = 3.;
  B[1] = 1.; B[3] = 5.;

  auto tensorA = tfctc::Tensor<float>({2, 2, 2}, A);
  auto tensorB = tfctc::Tensor<float>({2, 2}, B);
  auto tensorC = tfctc::Tensor<float>({2, 2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(1., tensorA, "abc", tensorB, "bd", 0., tensorC, "acd");
    requireAll(C, {
      0.1337, 4.7, 12.8, 4.63, 0.4011, 17.5, 53.4, 16.55
    });
  }

  memset(C, 0, 2 * 2 * 2 * sizeof(float));

  SUBCASE("different label order for C")
  {
    tfctc::contract(1., tensorA, "abc", tensorB, "bd", 0., tensorC, "adc");
    requireAll(C, {
      0.1337, 4.7, 0.4011, 17.5, 12.8, 4.63, 53.4, 16.55
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("(double) 2D . ID")
{
  double *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = 3.; A[2] = 4.;
  A[1] = 0.; A[3] = 1.;

  B[0] = 1.; B[2] = 0.;
  B[1] = 0.; B[3] = 1.;

  auto tensorA = tfctc::Tensor<double>({2, 2}, A);
  auto tensorB = tfctc::Tensor<double>({2, 2}, B);
  auto tensorC = tfctc::Tensor<double>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(1.0, tensorA, "ab", tensorB, "bc", 0., tensorC, "ac");
    requireAll(C, {
      3.,
      0.,
      4.,
      1.
    });
  }

  memset(C, 0, 2 * 2 * sizeof(double));

  SUBCASE("transposed")
  {
    tfctc::contract(1.0, tensorA, "ab", tensorB, "bc", 0., tensorC, "ca");
    requireAll(C, {
      3.,
      4.,
      0.,
      1.
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("(double) 2D . 2D => 2D")
{
  double *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = 3.; A[2] = 4.;
  A[1] = 1.; A[3] = 0.;

  B[0] = 1.; B[2] = 0.7;
  B[1] = 0.3; B[3] = 1.;

  auto tensorA = tfctc::Tensor<double>({2, 2}, A);
  auto tensorB = tfctc::Tensor<double>({2, 2}, B);
  auto tensorC = tfctc::Tensor<double>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(1., tensorA, "ab", tensorB, "bc", 0., tensorC, "ac");

    requireAll(C, {
      4.2,
      1.,
      6.1,
      0.7
    });
  }

  memset(C, 0, 2 * 2 * sizeof(double));

  SUBCASE("label order 2")
  {
    tfctc::contract(1., tensorA, "ab", tensorB, "cb", 0., tensorC, "ac");
    requireAll(C, {
      5.8,
      1.,
      4.9,
      0.3,
    });
  }

  memset(C, 0, 2 * 2 * sizeof(double));

  SUBCASE("label order 3")
  {
    tfctc::contract(1., tensorA, "ba", tensorB, "cb", 0., tensorC, "ac");
    requireAll(C, {
      3.7,
      4,
      1.9,
      1.2,
    });
  }

  memset(C, 0, 2 * 2 * sizeof(double));

  SUBCASE("label order 4")
  {
    tfctc::contract(1., tensorA, "ba", tensorB, "bc", 0., tensorC, "ac");
    requireAll(C, {
      3.3,
      4,
      3.1,
      2.8
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("float) 3D . 3D => 2D")
{
  double *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = 3.; A[2] = 3.;
  A[1] = 0;  A[3] = 1.7;
  
  A[4] = -1; A[6] = 1.7;
  A[5] = 3.; A[7] = 0.;


  B[0] = 0.33; B[2] = 0.;
  B[1] = 3;    B[3] = 4;

  B[4] = 1.;   B[6] = 4.;
  B[5] = 0.; B[7] = 0.;

  auto tensorA = tfctc::Tensor<double>({2, 2, 2}, A);
  auto tensorB = tfctc::Tensor<double>({2, 2, 2}, B);
  auto tensorC = tfctc::Tensor<double>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(1.0, tensorA, "abc", tensorB, "cbd", 0., tensorC, "ad");
    requireAll(C, {
      4.79,
      9,
      15,
      6.8
    });
  }

  memset(C, 0, 2 * 2 * sizeof(double));

  SUBCASE("transposed")
  {
    tfctc::contract(1., tensorA, "abc", tensorB, "cbd", 0., tensorC, "da");
    requireAll(C, {
      4.79,
      15,
      9,
      6.8
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("(float) 3D . 2D => 3D")
{
  double *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2 * 2);

  A[0] = 0.1337; A[2] = 0.;
  A[1] = 3.; A[3] = 1.7;

  A[4] = 5.3; A[6] = 7.5;
  A[5] = 3.3; A[7] = 1.33;

  B[0] = 1.; B[2] = 3.;
  B[1] = 1.; B[3] = 5.;

  auto tensorA = tfctc::Tensor<double>({2, 2, 2}, A);
  auto tensorB = tfctc::Tensor<double>({2, 2}, B);
  auto tensorC = tfctc::Tensor<double>({2, 2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(1., tensorA, "abc", tensorB, "bd", 0., tensorC, "acd");
    requireAll(C, {
      0.1337, 4.7, 12.8, 4.63, 0.4011, 17.5, 53.4, 16.55
    });
  }

  memset(C, 0, 2 * 2 * 2 * sizeof(double));

  SUBCASE("different label order for C")
  {
    tfctc::contract(1., tensorA, "abc", tensorB, "bd", 0., tensorC, "adc");
    requireAll(C, {
      0.1337, 4.7, 0.4011, 17.5, 12.8, 4.63, 53.4, 16.55
    });
  }

  free(A);
  free(B);
  free(C);
}