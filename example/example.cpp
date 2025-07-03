#include <complex>
#include <cstddef>
#include <memory>
#include "t1m/t1m.h"

int main() {
  constexpr std::size_t d = 10;
  constexpr std::size_t size_a = d * d * d;
  constexpr std::size_t size_b = d * d;
  constexpr std::size_t size_c = d * d * d;

  std::allocator<std::complex<float>> alloc{};

  std::complex<float>* data_a = alloc.allocate(size_a);
  std::complex<float>* data_b = alloc.allocate(size_b);
  std::complex<float>* data_c = alloc.allocate(size_c);

  // initialize values in column major

  t1m::tensor<std::complex<float>, 3> a({d, d, d}, data_a);
  t1m::tensor<std::complex<float>, 2> b({d, d}, data_b);
  t1m::tensor<std::complex<float>, 3> c({d, d, d}, data_c);

  t1m::contract(a, "abc", b, "bd", c, "acd");

  // ...

  alloc.deallocate(data_c, size_a);
  alloc.deallocate(data_b, size_b);
  alloc.deallocate(data_a, size_c);
}