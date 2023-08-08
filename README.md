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

In case you want refer to t1m as part of a research paper, please cite appropriately (pdf):

```text.bibtex
@thesis {
...
}
