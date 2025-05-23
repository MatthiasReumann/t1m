#pragma once

#include <blis/blis.h>
#include <cstdlib>

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

template <typename T, typename... Us>
requires std::is_same_v<T, float> void gemm_kernel(Us... args) {
  bli_sgemm_ukernel(args...);
}

template <typename T, typename... Us>
requires std::is_same_v<T, double> void gemm_kernel(Us... args) {
  bli_dgemm_ukernel(args...);
}

struct block_sizes {
  std::size_t MR;
  std::size_t NR;
  std::size_t KP;
  std::size_t MC;
  std::size_t KC;
  std::size_t NC;
};

template <typename T>
block_sizes get_block_sizes(const cntx_t* cntx, const num_t dt) {
  return {
      static_cast<std::size_t>(bli_cntx_get_blksz_def_dt(dt, BLIS_MR, cntx)),
      static_cast<std::size_t>(bli_cntx_get_blksz_def_dt(dt, BLIS_NR, cntx)),
      4,
      static_cast<std::size_t>(bli_cntx_get_blksz_def_dt(dt, BLIS_MC, cntx)),
      static_cast<std::size_t>(bli_cntx_get_blksz_def_dt(dt, BLIS_KC, cntx)),
      static_cast<std::size_t>(bli_cntx_get_blksz_def_dt(dt, BLIS_NC, cntx))};
}

template <typename T>
requires std::is_same_v<T, float> block_sizes
get_block_sizes(const cntx_t* cntx) {
  return get_block_sizes<T>(cntx, BLIS_FLOAT);
}

template <typename T>
requires std::is_same_v<T, double> block_sizes
get_block_sizes(const cntx_t* cntx) {
  return get_block_sizes<T>(cntx, BLIS_DOUBLE);
}

}  // namespace bli
}  // namespace t1m