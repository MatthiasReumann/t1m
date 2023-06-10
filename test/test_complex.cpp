#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tfctc.hpp"

#include <iostream>
#include <complex>

using FloatComplex = std::complex<float>;
using DoubleComplex = std::complex<double>;

inline void require(FloatComplex a, FloatComplex b)
{
  REQUIRE(a.real() == doctest::Approx(b.real()).epsilon(0.001));
  REQUIRE(a.imag() == doctest::Approx(b.imag()).epsilon(0.001));
}

inline void requireAll(FloatComplex *tensor, std::vector<FloatComplex> expected)
{
  for(int i = 0; i < expected.size(); i++)
  {
    require(tensor[i], expected[i]);
  }
}

TEST_CASE("(float) 2D . ID")
{
  FloatComplex *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = FloatComplex(3., 2.); A[2] = FloatComplex(0., 1.);
  A[1] = FloatComplex(0, -1.); A[3] = FloatComplex(1.);

  B[0] = FloatComplex(1.); B[2] = FloatComplex(0.);
  B[1] = FloatComplex(0);  B[3] = FloatComplex(1);

  auto tensorA = tfctc::Tensor<FloatComplex>({2, 2}, A);
  auto tensorB = tfctc::Tensor<FloatComplex>({2, 2}, B);
  auto tensorC = tfctc::Tensor<FloatComplex>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(tensorA, "ab", tensorB, "bc", tensorC, "ac");
    requireAll(C, {
      FloatComplex(3., 2.),
      FloatComplex(0., -1.),
      FloatComplex(0.,1.),
      FloatComplex(1.)
    });
  }

  SUBCASE("transposed")
  {
    tfctc::contract(tensorA, "ab", tensorB, "bc", tensorC, "ca");
    requireAll(C, {
      FloatComplex(3., 2.),
      FloatComplex(0., 1.),
      FloatComplex(0.,-1.),
      FloatComplex(1.)
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("float) 2D . 2D => 2D")
{
  FloatComplex *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = FloatComplex(3., 2.); A[2] = FloatComplex(0., 1.);
  A[1] = FloatComplex(0, -1.); A[3] = FloatComplex(1.);

  B[0] = FloatComplex(4.); B[2] = FloatComplex(0., 7.);
  B[1] = FloatComplex(-0.5, 0.5); B[3] = FloatComplex(3.3);

  auto tensorA = tfctc::Tensor<FloatComplex>({2, 2}, A);
  auto tensorB = tfctc::Tensor<FloatComplex>({2, 2}, B);
  auto tensorC = tfctc::Tensor<FloatComplex>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(tensorA, "ab", tensorB, "bc", tensorC, "ac");

    requireAll(C, {
      FloatComplex(11.5, 7.5),
      FloatComplex(-0.5, -3.5),
      FloatComplex(-14, 24.3),
      FloatComplex(10.3, 0.)
    });
  }

  memset(C, 0, 2 * 2 * sizeof(FloatComplex));

  SUBCASE("")
  {
    tfctc::contract(tensorA, "ab", tensorB, "cb", tensorC, "ac");
    requireAll(C, {
      FloatComplex(5, 8),
      FloatComplex(0, 3.),
      FloatComplex(-2.5, 3.8),
      FloatComplex(3.8, 0.5)
    });
  }

  memset(C, 0, 2 * 2 * sizeof(FloatComplex));

  SUBCASE("")
  {
    tfctc::contract(tensorA, "ba", tensorB, "cb", tensorC, "ac");
    requireAll(C, {
      FloatComplex(19., 8.),
      FloatComplex(0.,11.),
      FloatComplex(-2.5, -2.8),
      FloatComplex(2.8, -0.5)
    });
  }

  memset(C, 0, 2 * 2 * sizeof(FloatComplex));

  SUBCASE("")
  {
    tfctc::contract(tensorA, "ba", tensorB, "bc", tensorC, "ac");
    requireAll(C, {
      FloatComplex(12.5, 8.5),
      FloatComplex(-0.5, 4.5),
      FloatComplex(-14, 17.7),
      FloatComplex(-3.7, 0.)
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("float) 3D . 3D => 2D")
{
  FloatComplex *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2);

  A[0] = FloatComplex(0., 1.); A[2] = FloatComplex(0.);
  A[1] = FloatComplex(0.);     A[3] = FloatComplex(0., 1.);
  
  A[4] = FloatComplex(0., -1.); A[6] = FloatComplex(0);
  A[5] = FloatComplex(0.);      A[7] = FloatComplex(0., -1.);


  B[0] = FloatComplex(0.33); B[2] = FloatComplex(1., 1.);
  B[1] = FloatComplex(0.);   B[3] = FloatComplex(0., 1.);

  B[4] = FloatComplex(0.);      B[6] = FloatComplex(0.47);
  B[5] = FloatComplex(0., 0.7); B[7] = FloatComplex(0.1337);

  auto tensorA = tfctc::Tensor<FloatComplex>({2, 2, 2}, A);
  auto tensorB = tfctc::Tensor<FloatComplex>({2, 2, 2}, B);
  auto tensorC = tfctc::Tensor<FloatComplex>({2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(tensorA, "abc", tensorB, "cbd", tensorC, "ad");
    requireAll(C, {
      FloatComplex(0., 0.33),
      FloatComplex(0., 1.),
      FloatComplex(0.7, 0.),
      FloatComplex(0., 0.3363)
    });
  }

  memset(C, 0, 2 * 2 * sizeof(FloatComplex));

  SUBCASE("transposed")
  {
    tfctc::contract(tensorA, "abc", tensorB, "cbd", tensorC, "da");
    requireAll(C, {
      FloatComplex(0., 0.33),
      FloatComplex(0.7, 0.),
      FloatComplex(0., 1.),
      FloatComplex(0., 0.3363)
    });
  }

  free(A);
  free(B);
  free(C);
}

TEST_CASE("float) 3D . 2D => 3D")
{
  FloatComplex *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2 * 2);

  A[0] = FloatComplex(0., 1.); A[2] = FloatComplex(0.);
  A[1] = FloatComplex(0.); A[3] = FloatComplex(0., 1.);

  A[4] = FloatComplex(0., -1.); A[6] = FloatComplex(0);
  A[5] = FloatComplex(0.); A[7] = FloatComplex(0., -1.);

  B[0] = FloatComplex(4.); B[2] = FloatComplex(0., 7.);
  B[1] = FloatComplex(-0.5, 0.5); B[3] = FloatComplex(3.3);

  auto tensorA = tfctc::Tensor<FloatComplex>({2, 2, 2}, A);
  auto tensorB = tfctc::Tensor<FloatComplex>({2, 2}, B);
  auto tensorC = tfctc::Tensor<FloatComplex>({2, 2, 2}, C);

  SUBCASE("standard")
  {
    tfctc::contract(tensorA, "abc", tensorB, "bd", tensorC, "acd");
    requireAll(C, {
      FloatComplex(0, 4),
      FloatComplex(-0.5, -0.5),
      FloatComplex(0,-4),
      FloatComplex(0.5, 0.5),
      FloatComplex(-7.),
      FloatComplex(0., 3.3),
      FloatComplex(7.),
      FloatComplex(0., -3.3),
    });
  }

  memset(C, 0, 2 * 2 * sizeof(FloatComplex));

  SUBCASE("different label order for C")
  {
    tfctc::contract(tensorA, "abc", tensorB, "bd", tensorC, "adc");
    requireAll(C, {
      FloatComplex(0, 4),
      FloatComplex(-0.5, -0.5),
      FloatComplex(-7),
      FloatComplex(0., 3.3),
      FloatComplex(0.,-4.),
      FloatComplex(0.5,0.5),
      FloatComplex(7.),
      FloatComplex(0., -3.3),
    });
  }

  free(A);
  free(B);
  free(C);
}