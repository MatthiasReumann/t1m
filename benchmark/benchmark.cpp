#include <chrono>
#include <numeric>
#include <print>
#include "t1m/bli/mappings.h"
#include "t1m/t1m.h"

constexpr std::size_t REPEATS = 10;

namespace {
template <std::size_t ndim>
std::size_t multiply(const std::array<std::size_t, ndim>& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<std::size_t>());
}

template <class T, std::size_t ndim_a, std::size_t ndim_b, std::size_t ndim_c>
void explicit_benchmark(const std::array<std::size_t, ndim_a>& dims_a,
                        const std::string& labels_a,
                        const std::array<std::size_t, ndim_b>& dims_b,
                        const std::string& labels_b,
                        const std::array<std::size_t, ndim_c>& dims_c,
                        const std::string& labels_c) {
  std::allocator<T> alloc{};

  constexpr t1m::memory_layout layout = t1m::col_major;

  const T zero = T(0);

  const std::size_t nelems_a = multiply(dims_a);
  const std::size_t nelems_b = multiply(dims_b);
  const std::size_t nelems_c = multiply(dims_c);

  T* data_a = alloc.allocate(nelems_a);
  T* data_b = alloc.allocate(nelems_b);
  T* data_c = alloc.allocate(nelems_c);

  t1m::bli::randv<T>(nelems_a, data_a, 1);
  t1m::bli::randv<T>(nelems_b, data_b, 1);

  t1m::tensor<T, ndim_a> a{dims_a, data_a, layout};
  t1m::tensor<T, ndim_b> b{dims_b, data_b, layout};
  t1m::tensor<T, ndim_c> c{dims_c, data_c, layout};

  std::print("{}-{}-{},", labels_a, labels_b, labels_c);

  std::array<double, REPEATS> time{};
  for (std::size_t i = 0; i < time.size(); ++i) {
    t1m::bli::setv<T>(BLIS_NO_CONJUGATE, nelems_c, &zero, data_c, 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    t1m::contract<T>(1., a, labels_a, b, labels_b, 0., c, labels_c);
    auto t1 = std::chrono::high_resolution_clock::now();
    time[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  }
  std::println("{}", *std::min_element(time.begin(), time.end()));

  alloc.deallocate(data_c, nelems_c);
  alloc.deallocate(data_b, nelems_b);
  alloc.deallocate(data_a, nelems_a);
}
}  // namespace

template <typename T>
void suite() {
  std::println("contraction,min");
  // clang-format off
  explicit_benchmark<T, 5, 2, 5>(
    {24, 16, 16, 24, 16}, "efbad", 
    {12, 16}, "cf", 
    {24, 16, 12, 16, 24}, "abcde");
  explicit_benchmark<T, 5, 2, 5>(
    {24, 16, 16, 24, 16}, "efcad", 
    {12, 16}, "bf", 
    {24, 12, 16, 16, 24}, "abcde");
  explicit_benchmark<T, 4, 2, 4>(
    {36, 36, 36, 36}, "dbea", 
    {36, 12}, "ec",
    {36, 36, 12, 36}, "abcd");
  explicit_benchmark<T, 5, 2, 5>(
    {24, 16, 16, 24, 24}, "ecbfa", 
    {24, 12}, "fd", 
    {24, 16, 16, 12, 24}, "abcde");
  explicit_benchmark<T, 4, 2, 4>(
    {36, 36, 36, 36}, "deca", 
    {12, 36}, "be",
    {36, 12, 36, 36}, "abcd");
   explicit_benchmark<T, 3, 2, 3>(
    {156, 156, 156}, "bda", 
    {156, 12}, "dc",
    {156, 156, 12}, "abc");
  explicit_benchmark<T, 4, 2, 4>(
    {36, 36, 36, 36}, "ebad", 
    {12, 36}, "ce",
    {36, 36, 12, 36}, "abcd");
  explicit_benchmark<T, 4, 4, 6>(
    {12, 8, 12, 12}, "dega", 
    {12, 8, 8, 8}, "gfbc",
    {12, 8, 8, 12, 8, 8}, "abcdef");
  explicit_benchmark<T, 4, 4, 6>(
    {12, 8, 12, 8}, "dfgb", 
    {12, 8, 12, 8}, "geac",
    {12, 8, 8, 12, 8, 8}, "abcdef");
  explicit_benchmark<T, 4, 4, 6>(
    {12, 8, 12, 8}, "degb", 
    {12, 8, 12, 8}, "gfac",
    {12, 8, 8, 12, 8, 8}, "abcdef");
  explicit_benchmark<T, 4, 4, 6>(
    {12, 8, 12, 8}, "degc", 
    {12, 8, 12, 8}, "gfab",
    {12, 8, 8, 12, 8, 8}, "abcdef");
  explicit_benchmark<T, 3, 2, 3>(
    {156, 148, 156}, "dca", 
    {12, 156}, "bd",
    {156, 12, 148}, "abc");
  explicit_benchmark<T, 2, 4, 4>(
    {36, 36}, "ea", 
    {36, 36, 36, 36}, "ebcd",
    {36, 36, 36, 36}, "abcd");
  explicit_benchmark<T, 2, 4, 4>(
    {36, 36}, "eb", 
    {36, 36, 36, 36}, "aecd",
    {36, 36, 36, 36}, "abcd");
  explicit_benchmark<T, 2, 4, 4>(
    {36, 36}, "ec", 
    {36, 36, 36, 36}, "abed",
    {36, 36, 36, 36}, "abcd");
  explicit_benchmark<T, 4, 3, 3>(
    {36, 36, 36, 36}, "adec", 
    {36, 36, 36}, "ebd",
    {36, 36, 36}, "abc");
  explicit_benchmark<T, 3, 3, 2>(
    {156, 156, 156}, "cad", 
    {156, 156, 148}, "dcb",
    {156, 148}, "ab");
  explicit_benchmark<T, 3, 3, 2>(
    {156, 148, 156}, "acd", 
    {156, 148, 148}, "dbc",
    {156, 148}, "ab");
  explicit_benchmark<T, 3, 2, 3>(
    {156, 148, 148}, "acd", 
    {148, 156}, "db",
    {156, 156, 148}, "abc");
  explicit_benchmark<T, 3, 2, 3>(
    {156, 148, 148}, "adc", 
    {156, 148}, "bd",
    {156, 156, 148}, "abc");
   explicit_benchmark<T, 2, 2, 2>(
    {2568, 2568}, "ac", 
    {2568, 2560}, "cb",
    {2568, 2560}, "ab");
   explicit_benchmark<T, 4, 4, 4>(
    {36, 36, 36, 36}, "aebf", 
    {36, 36, 36, 36}, "fdec",
    {36, 36, 36, 36}, "abcd");
  explicit_benchmark<T, 4, 4, 4>(
    {36, 36, 36, 36}, "eafd", 
    {36, 36, 36, 36}, "fbec",
    {36, 36, 36, 36}, "abcd");
  explicit_benchmark<T, 4, 4, 4>(
    {36, 36, 36, 36}, "aebf", 
    {36, 36, 36, 36}, "dfce",
    {36, 36, 36, 36}, "abcd");
  // clang-format on
}

int main() {
  suite<float>();
  suite<double>();
}