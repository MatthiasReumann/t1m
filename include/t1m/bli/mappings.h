#pragma once

#include <blis.h>
#include <complex>
#include <cstdlib>
#include <type_traits>

namespace t1m {
namespace bli {

/**
 * @brief Call bli_?setv based on the templated datatype.
 */
template <typename T, typename... ArgsT> void setv(ArgsT... args) {
  if constexpr (std::is_same_v<T, float>) {
    bli_ssetv(args...);
  } else if constexpr (std::is_same_v<T, double>) {
    bli_dsetv(args...);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    bli_csetv(args...);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    bli_zsetv(args...);
  }
}

/**
 * @brief Call bli_?randv based on the templated datatype.
 */
template <typename T, typename... ArgsT> void randv(ArgsT... args) {
  if constexpr (std::is_same_v<T, float>) {
    bli_srandv(args...);
  } else if constexpr (std::is_same_v<T, double>) {
    bli_drandv(args...);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    bli_crandv(args...);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    bli_zrandv(args...);
  }
}

/**
 * @brief Call bli_?normv based on the templated datatype.
 */
template <typename T, typename... ArgsT> void normv(ArgsT... args) {
  if constexpr (std::is_same_v<T, float>) {
    bli_snormfv(args...);
  } else if constexpr (std::is_same_v<T, double>) {
    bli_dnormfv(args...);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    bli_cnormfv(args...);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    bli_znormfv(args...);
  }
}

/**
 * @brief Call bli_?axpym based on the templated datatype.
 */
template <typename T, typename... ArgsT> void axpym(ArgsT... args) {
  if constexpr (std::is_same_v<T, float>) {
    bli_saxpym(args...);
  } else if constexpr (std::is_same_v<T, double>) {
    bli_daxpym(args...);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    bli_caxpym(args...);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    bli_zaxpym(args...);
  }
}

/**
 * @brief Call bli_?gemm based on the templated datatype.
 */
template <typename T, typename... ArgsT> void gemm(ArgsT... args) {
  if constexpr (std::is_same_v<T, float>) {
    bli_sgemm(args...);
  } else if constexpr (std::is_same_v<T, double>) {
    bli_dgemm(args...);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    bli_cgemm(args...);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    bli_zgemm(args...);
  }
}

/**
 * @brief Call bli_?gemm_ukernel based on the templated datatype.
 * @details Due to the 1M method, the kernel used for complex 
 *          operations is real also.
 */
template <typename T, typename... ArgsT> void gemm_kernel(ArgsT... args) {
  if constexpr (std::is_same_v<T, float> ||
                std::is_same_v<T, std::complex<float>>) {
    bli_sgemm_ukernel(args...);
  } else if constexpr (std::is_same_v<T, double> ||
                       std::is_same_v<T, std::complex<double>>) {
    bli_dgemm_ukernel(args...);
  }
}

/**
 * @brief Block sizes used for the five loops around the microkernel 
 *        algorithm as well as for calculating the block scatter vectors.
 */
struct block_sizes {
  std::size_t MR;
  std::size_t NR;
  std::size_t KP;
  std::size_t MC;
  std::size_t KC;
  std::size_t NC;
};

/**
 * @brief Return BLIS block sizes for a given datatype.
 */
template <typename T>
block_sizes get_block_sizes(const cntx_t* cntx, const num_t dt) {
  struct block_sizes bs;
  bs.MR = bli_cntx_get_blksz_def_dt(dt, BLIS_MR, cntx);
  bs.NR = bli_cntx_get_blksz_def_dt(dt, BLIS_NR, cntx);
  bs.KP = 4;
  bs.MC = bli_cntx_get_blksz_def_dt(dt, BLIS_MC, cntx);
  bs.KC = bli_cntx_get_blksz_def_dt(dt, BLIS_KC, cntx);
  bs.NC = bli_cntx_get_blksz_def_dt(dt, BLIS_NC, cntx);
  return bs;
}

/**
 * @brief Return block sizes based on the templated datatype.
 * @details Due to the 1M method, the block sizes used for complex 
 *          operations equal the real ones.
 */
template <typename T> block_sizes get_block_sizes(const cntx_t* cntx) {
  if constexpr (std::is_same_v<T, float> ||
                std::is_same_v<T, std::complex<float>>) {
    return get_block_sizes<T>(cntx, BLIS_FLOAT);
  } else if (std::is_same_v<T, double> ||
             std::is_same_v<T, std::complex<double>>) {
    return get_block_sizes<T>(cntx, BLIS_DOUBLE);
  }
}
}  // namespace bli
}  // namespace t1m