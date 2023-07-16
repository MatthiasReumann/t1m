# t1m

## Requirements

- BLIS Library ([URL](https://github.com/flame/blis))
- MArray Library ([URL](https://github.com/devinamatthews/marray))


## API 

```cpp
namespace tfctc
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

### Example

```cpp
#include <complex>
#include "tfctc.hpp"

int main() 
{
  std::complex<float> *A = nullptr, *B = nullptr, *C = nullptr;
  tfctc::utils::alloc_aligned(&A, 2 * 2 * 2);
  tfctc::utils::alloc_aligned(&B, 2 * 2);
  tfctc::utils::alloc_aligned(&C, 2 * 2 * 2);
  
  // initialize values in column major

  auto tensorA = tfctc::Tensor<std::complex<float>>({2, 2, 2}, A);
  auto tensorB = tfctc::Tensor<std::complex<float>>({2, 2}, B);
  auto tensorC = tfctc::Tensor<std::complex<float>>({2, 2, 2}, C);

  tfctc::contract(tensorA, "abc", tensorB, "bd", tensorC, "acd");
  
  // work with C or tensorC
  
  free(A);
  free(B);
  free(C);
}
```

## Citation

In case you want refer to TFCTC as part of a research paper, please cite appropriately (pdf):

```text.bibtex
@thesis {
...
}
