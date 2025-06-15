#include <complex>
#include <cstddef>
#include <memory>
#include "t1m/t1m.h"
#include "t1m/tensor.h"

int main() {
  constexpr std::size_t d = 10;
  constexpr t1m::memory_layout layout = t1m::memory_layout::col_major;

  std::allocator<std::complex<float>> alloc{};

  std::complex<float>* data_a = alloc.allocate(d * d * d);
  std::complex<float>* data_b = alloc.allocate(d * d);
  std::complex<float>* data_c = alloc.allocate(d * d * d);

  // initialize values in column major

  t1m::tensor<std::complex<float>, 3> a{{d, d, d}, data_a, layout};
  t1m::tensor<std::complex<float>, 2> b{{d, d}, data_b, layout};
  t1m::tensor<std::complex<float>, 3> c{{d, d, d}, data_c, layout};

  t1m::contract(a, "abc", b, "bd", c, "acd");

  // ...

  alloc.deallocate(data_c, 2 * 2 * 2);
  alloc.deallocate(data_b, 2 * 2);
  alloc.deallocate(data_a, 2 * 2 * 2);
}