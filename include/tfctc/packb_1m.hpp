#pragma once

#include <complex>
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename U>
    void pack_1m_ab(U* ptr_a, const std::complex<U> c, const dim_t MR)
    {
      ptr_a[0] = c.real(); ptr_a[MR] = -c.imag();
      ptr_a[1] = c.imag(); ptr_a[MR + 1] = c.real();
    }

    template <typename U>
    void pack_1m_bb(U* ptr_b, const std::complex<U> c, const dim_t NR)
    {
      ptr_b[0] = c.real();
      ptr_b[NR] = c.imag();
    }
  };
};
