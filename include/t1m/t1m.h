#pragma once

#include <complex>
#include "internal/utils.h"
#include "t1m/tensor.h"
#include "tensor.hpp"

namespace t1m {

void contract(Tensor<std::complex<float>> A, std::string labelsA,
              Tensor<std::complex<float>> B, std::string labelsB,
              Tensor<std::complex<float>> C, std::string labelsC);

void contract(Tensor<std::complex<double>> A, std::string labelsA,
              Tensor<std::complex<double>> B, std::string labelsB,
              Tensor<std::complex<double>> C, std::string labelsC);

void contract(float alpha, Tensor<float> A, std::string labelsA,
              Tensor<float> B, std::string labelsB, float beta, Tensor<float> C,
              std::string labelsC);

void contract(double alpha, Tensor<double> A, std::string labelsA,
              Tensor<double> B, std::string labelsB, double beta,
              Tensor<double> C, std::string labelsC);
};  // namespace t1m
