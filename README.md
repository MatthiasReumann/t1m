# t1m

A header-only library for tensor contractions. The transposition-free tensor contraction algorithm fuses the [**T**BLIS](https://github.com/devinamatthews/tblis) approach and the [**1M** method](https://www.cs.utexas.edu/users/flame/pubs/blis6_toms_rev2.pdf) for complex-valued matrix-matrix multiplication.

## Requirements

- [BLIS](https://github.com/flame/blis)

## Usage

```cpp
#include <complex>
#include <cstddef>
#include <memory>
#include "t1m/t1m.h"
#include "t1m/tensor.h"

int main() {
  constexpr std::size_t d = 10;
  constexpr std::size_t size_a = d * d * d;
  constexpr std::size_t size_b = d * d;
  constexpr std::size_t size_c = d * d * d;
  constexpr t1m::memory_layout layout = t1m::memory_layout::col_major;

  std::allocator<std::complex<float>> alloc{};

  std::complex<float>* data_a = alloc.allocate(size_a);
  std::complex<float>* data_b = alloc.allocate(size_b);
  std::complex<float>* data_c = alloc.allocate(size_c);

  // initialize values in column major

  t1m::tensor<std::complex<float>, 3> a{{d, d, d}, data_a, layout};
  t1m::tensor<std::complex<float>, 2> b{{d, d}, data_b, layout};
  t1m::tensor<std::complex<float>, 3> c{{d, d, d}, data_c, layout};

  t1m::contract(a, "abc", b, "bd", c, "acd");

  // ...

  alloc.deallocate(data_c, size_a);
  alloc.deallocate(data_b, size_b);
  alloc.deallocate(data_a, size_c);
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
