#pragma once

#include <complex>
#include "utils.hpp"
#include "std_ext.hpp"
#include "gemm_context.hpp"
#include "scatter_matrix.hpp"
#include "block_scatter_matrix.hpp"
#include "packing.hpp"
#include "packing_1m.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename T>
    void gemm_1m(const gemm_context_1m<std::complex<T>, T>* ctx)
    {
      auto A = ctx->A;
      auto B = ctx->B;
      auto C = ctx->C;

      const dim_t NC = ctx->NC;
      const dim_t KC = ctx->KC;
      const dim_t MC = ctx->MC;
      const dim_t NR = ctx->NR;
      const dim_t MR = ctx->MR;
      const dim_t KP = ctx->KP;

      const size_t M = A->row_size();
      const size_t K = A->col_size();
      const size_t N = B->col_size();

      size_t x, y;
      dim_t m1, n1, k1, m, n, mreal, kreal;
      inc_t rsc = 1, csc;

      T* buf = nullptr;
      T* A_tilde = nullptr; // A in G^{MC x KC}
      T* A_tilde_base = nullptr;

      T* B_tilde = nullptr; // B in G^{KC x NC}
      T* B_tilde_base = nullptr;

      T* C_tilde = nullptr; // C in G^{MC x NC}

      tfctc::utils::alloc_aligned<T>(&buf, MC * KC + KC * NC + MC * NC);

      A_tilde = A_tilde_base = buf;
      B_tilde = B_tilde_base = buf + MC * KC;
      C_tilde = buf + MC * KC + KC * NC;

      for (size_t j_c = 0; j_c < N; j_c += NC)
      {
        n1 = tfctc::std_ext::min(NC, static_cast<dim_t>(N - j_c));

        for (size_t p_c = 0; p_c < K; p_c += KC / 2)
        {
          k1 = tfctc::std_ext::min(KC / 2, static_cast<dim_t>(K - p_c));
          kreal = k1 * 2;

          memset(B_tilde, 0, KC * NC * sizeof(T));
          internal::pack_1m_b(B, B_tilde, p_c, j_c, k1, n1, NR, KP);

          // B is now row-major packed into a KC * NC buffer
          // with the specialized format such that each sliver
          // has stride NR

          for (size_t i_c = 0; i_c < M; i_c += MC / 2)
          {
            m1 = tfctc::std_ext::min(MC / 2, static_cast<dim_t>(M - i_c));
            mreal = m1 * 2;

            memset(A_tilde, 0, MC * KC * sizeof(T));
            internal::pack_1m_a(A, A_tilde, i_c, p_c, m1, k1, MR, KP);

            // A is now column-major packed into a MC * KC buffer
            // with the specialized format such that each sliver
            // has stride MR

            // Now treat everything as real-valued:
            // Use NR, MR as with real-valued mm
            for (size_t j_r = 0; j_r < n1; j_r += NR)
            {
              y = j_c + j_r;
              n = tfctc::std_ext::min(NR, static_cast<dim_t>(n1 - j_r));
              csc = C->col_stride_in_block(y / NR);

              for (size_t i_r = 0; i_r < mreal; i_r += MR)
              {
                m = tfctc::std_ext::min(MR, static_cast<dim_t>(mreal - i_r));
                x = i_c + (i_r / 2);
                rsc = C->row_stride_in_block(x / MR);

                ctx->kernel(m, n, kreal,
                  ctx->alpha,
                  A_tilde,
                  B_tilde,
                  ctx->beta,
                  C_tilde, 1, m,
                  nullptr,
                  ctx->cntx);

                if (rsc > 0 && csc > 0)
                  internal::unpack_1m_c_cont(C, C_tilde, x, y, m, n, rsc, csc);
                else
                  internal::unpack_1m_c(C, C_tilde, x, y, m, n);

                A_tilde += MR * kreal;
              }
              B_tilde += kreal * NR;

              A_tilde = A_tilde_base;
            }
            B_tilde = B_tilde_base;
          }
        }
      }

      free(buf);
    }

    template <typename T>
    void gemm(const gemm_context<T>* ctx)
    {
      BlockScatterMatrix<T>* A = ctx->A;
      BlockScatterMatrix<T>* B = ctx->B;
      BlockScatterMatrix<T>* C = ctx->C;

      const dim_t NC = ctx->NC;
      const dim_t KC = ctx->KC;
      const dim_t MC = ctx->MC;
      const dim_t NR = ctx->NR;
      const dim_t MR = ctx->MR;
      const dim_t KP = ctx->KP;

      const size_t M = A->row_size();
      const size_t K = A->col_size();
      const size_t N = B->col_size();

      T* buf = nullptr;
      T* A_tilde = nullptr; // A in G^{MC x KC}
      T* A_tilde_base = nullptr;

      T* B_tilde = nullptr; // B in G^{KC x NC}
      T* B_tilde_base = nullptr;

      T* C_tilde = nullptr; // C in G^{MC x NC}
      T* C_tilde_base = nullptr;

      tfctc::utils::alloc_aligned<T>(&buf, MC * KC + KC * NC + MC * NC);

      A_tilde = A_tilde_base = buf;
      B_tilde = B_tilde_base = buf + MC * KC;
      C_tilde = C_tilde_base = buf + MC * KC + KC * NC;

      size_t x, y;
      dim_t m1, n1, k1, m, n;
      inc_t rsc = 1, csc;

      for (size_t j_c = 0; j_c < N; j_c += NC)
      {
        n1 = tfctc::std_ext::min(NC, static_cast<dim_t>(N - j_c));

        for (size_t p_c = 0; p_c < K; p_c += KC)
        {
          k1 = tfctc::std_ext::min(KC, static_cast<dim_t>(K - p_c));

          memset(B_tilde, 0, KC * NC * sizeof(T));
          internal::pack_b(B, B_tilde, p_c, j_c, k1, n1, NR, KP);

          for (size_t i_c = 0; i_c < M; i_c += MC)
          {
            m1 = tfctc::std_ext::min(MC, static_cast<dim_t>(M - i_c));

            memset(A_tilde, 0, MC * KC * sizeof(T));
            internal::pack_a(A, A_tilde, i_c, p_c, m1, k1, MR, KP);

            for (size_t j_r = 0; j_r < n1; j_r += NR)
            {
              n = tfctc::std_ext::min(NR, static_cast<dim_t>(n1 - j_r));
              y = j_c + j_r;
              csc = C->col_stride_in_block(y / NR);

              for (size_t i_r = 0; i_r < m1; i_r += MR)
              {
                m = tfctc::std_ext::min(MR, static_cast<dim_t>(m1 - i_r));
                x = i_c + i_r;
                rsc = C->row_stride_in_block(x / MR);

                if (rsc > 0 && csc > 0)
                {
                  ctx->kernel(m, n, k1,
                    ctx->alpha,
                    A_tilde,
                    B_tilde,
                    ctx->beta,
                    C->pointer_at_loc(x, y), rsc, csc,
                    nullptr,
                    ctx->cntx);
                }
                else {
                  ctx->kernel(m, n, k1,
                    ctx->alpha,
                    A_tilde,
                    B_tilde,
                    ctx->beta,
                    C_tilde, 1, m,
                    nullptr,
                    ctx->cntx);

                  internal::unpack_c_scat(C, C_tilde, x, y, m, n);
                }

                A_tilde += MR * k1;
              }
              B_tilde += k1 * NR;

              A_tilde = A_tilde_base;
            }
            B_tilde = B_tilde_base;
          }
        }
      }

      free(buf);
    }
  }
};
