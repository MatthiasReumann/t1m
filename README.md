# t1m

Fusion of the [**T**BLIS](https://github.com/devinamatthews/tblis) approach and the [**1M** Method](https://www.cs.utexas.edu/users/flame/pubs/blis6_toms_rev2.pdf) for complex Matrix-Matrix Multiplication to achieve complex Tensor Contractions. 

## Requirements

- BLIS Library ([URL](https://github.com/flame/blis))
- MArray Library ([URL](https://github.com/devinamatthews/marray))


## API 

```cpp
namespace t1m
{
  void contract(Tensor<std::complex<float>> A, std::string labelsA,
                Tensor<std::complex<float>> B, std::string labelsB,
                Tensor<std::complex<float>> C, std::string labelsC);

  void contract(Tensor<std::complex<double>> A, std::string labelsA,
                Tensor<std::complex<double>> B, std::string labelsB,
                Tensor<std::complex<double>> C, std::string labelsC);

  void contract(float alpha, Tensor<float> A, std::string labelsA,
                Tensor<float> B, std::string labelsB,
                float beta, Tensor<float> C, std::string labelsC);

  void contract(double alpha, Tensor<double> A, std::string labelsA,
                Tensor<double> B, std::string labelsB,
                double beta, Tensor<double> C, std::string labelsC);
};
```

### Multithreading 

The `t1m` library supports OpenMP. The number of threads can be specified with the environment variable `OMP_NUM_THREADS`.

### Example

```cpp
#include <complex>
#include "t1m.hpp"

int main() 
{
  std::complex<float> *A = nullptr, *B = nullptr, *C = nullptr;
  t1m::utils::alloc_aligned(&A, 2 * 2 * 2);
  t1m::utils::alloc_aligned(&B, 2 * 2);
  t1m::utils::alloc_aligned(&C, 2 * 2 * 2);
  
  // initialize values in column major

  auto tensorA = t1m::Tensor<std::complex<float>>({2, 2, 2}, A);
  auto tensorB = t1m::Tensor<std::complex<float>>({2, 2}, B);
  auto tensorC = t1m::Tensor<std::complex<float>>({2, 2, 2}, C);

  t1m::contract(tensorA, "abc", tensorB, "bd", tensorC, "acd");
  
  // work with C or tensorC
  
  free(A);
  free(B);
  free(C);
}
```

## Citation

In case you want refer to `t1m` as part of a research paper, please cite appropriately ([pdf](https://mediatum.ub.tum.de/download/1718165/1718165.pdf)):

```text.bibtex
@thesis {t1m2023,
  author = {Matthias Reumann},
  title = {Transpose-Free Contraction of Complex Tensors},
  year = {2023},
  school = {Technical University of Munich},
  month = {Aug},
  language = {en},
  abstract = {Tensor Contraction (TC) is the operation that connects tensors in a Tensor Network (TN). Many scientific applications rely on efficient algorithms for the contraction of large tensors. In this thesis, we aim to develop a transposition-free TC algorithm for complex tensors. Our algorithm fuses high-performance General Matrix-Matrix Multiplication (GEMM), the 1M method for achieving complex with real-valued GEMM, and the Block-Scatter layout for tensors. Consequently, we give an elaborate overview of each. A benchmark for a series of contractions shows that our implementation can compete with the performance of state-of-the-art TC libraries.},
}
