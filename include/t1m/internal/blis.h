#include <blis/blis.h>

namespace t1m {
namespace bli {
    template <typename T, typename... Us>
requires std::is_same_v<T, float> void setv(Us... args) {
  bli_ssetv(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, double> void setv(Us... args) {
  bli_dsetv(args...);
}

 template <typename T, typename... Us>
requires std::is_same_v<T, float> void randv(Us... args) {
  bli_srandv(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, double> void randv(Us... args) {
  bli_drandv(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, float> void normv(Us... args) {
  bli_snormfv(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, double> void normv(Us... args) {
  bli_dnormfv(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, float> void axpym(Us... args) {
  bli_saxpym(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, double> void axpym(Us... args) {
  bli_daxpym(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, float> void gemm(Us... args) {
  bli_sgemm(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, double> void gemm(Us... args) {
  bli_dgemm(args...);
}
}  // namespace bli
}  // namespace t1m