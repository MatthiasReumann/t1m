#include <complex>
#include <memory>
#include "t1m/t1m.h"
#include "t1m/tensor.h"

int main() {
  constexpr t1m::memory_layout layout = t1m::memory_layout::col_major;

  std::allocator<std::complex<float>> alloc{};

  std::complex<float>* data_a = alloc.allocate(2 * 2 * 2);
  std::complex<float>* data_b = alloc.allocate(2 * 2);
  std::complex<float>* data_c = alloc.allocate(2 * 2 * 2);

  // initialize values in column major

  t1m::tensor<std::complex<float>, 3> a{{2, 2, 2}, data_a, layout};
  t1m::tensor<std::complex<float>, 2> b{{2, 2}, data_b, layout};
  t1m::tensor<std::complex<float>, 3> c{{2, 2, 2}, data_c, layout};

  t1m::contract(a, "abc", b, "bd", c, "acd");

  // ...

  alloc.deallocate(data_c, 2 * 2 * 2);
  alloc.deallocate(data_b, 2 * 2);
  alloc.deallocate(data_a, 2 * 2 * 2);
}