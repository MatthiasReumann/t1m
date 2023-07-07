#pragma once

#include <complex>
#include "utils.hpp"
#include "std_ext.hpp"
#include "gemm_context.hpp"
#include "scatter_matrix.hpp"
#include "block_scatter_matrix.hpp"
#include "packm.hpp"
#include "packm_1m.hpp"
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

      size_t off_i, off_j;
      dim_t mc_m_complex, nc_n, kc_k_complex, mc_m_real, k, m, n;
      inc_t rsc = 1, csc;

      T* workspace = nullptr;
      T* a_packed = nullptr; // A in G^{MC x KC}
      T* a_packed_base = nullptr;

      T* b_packed = nullptr; // B in G^{KC x NC}
      T* b_packed_base = nullptr;

      T* c_result = nullptr; // C in G^{MC x NC}

      utils::alloc_aligned<T>(&workspace, MC * KC + KC * NC + MC * NC);

      a_packed = a_packed_base = workspace;
      b_packed = b_packed_base = workspace + MC * KC;
      c_result = workspace + MC * KC + KC * NC;

      for (size_t j_c = 0; j_c < N; j_c += NC)
      {
        nc_n = std_ext::min(NC, static_cast<dim_t>(N - j_c));

        for (size_t p_c = 0; p_c < K; p_c += KC / 2)
        {
          kc_k_complex = std_ext::min(KC / 2, static_cast<dim_t>(K - p_c));
          k = kc_k_complex * 2;

          memset(b_packed, 0, KC * NC * sizeof(T));
          pack_1m_b(B, b_packed, p_c, j_c, kc_k_complex, nc_n, NR, KP);

          // B is now row-major packed into a KC * NC buffer
          // with the specialized format such that each sliver
          // has stride NR
          for (size_t i_c = 0; i_c < M; i_c += MC / 2)
          {
            mc_m_complex = std_ext::min(MC / 2, static_cast<dim_t>(M - i_c));
            mc_m_real = mc_m_complex * 2;

            memset(a_packed, 0, MC * KC * sizeof(T));
            pack_1m_a(A, a_packed, i_c, p_c, mc_m_complex, kc_k_complex, MR, KP);

            // A is now column-major packed into a MC * KC buffer
            // with the specialized format such that each sliver
            // has stride MR

            // Now treat everything as real-valued:
            // Use NR, MR as with real-valued mm
            for (size_t j_r = 0; j_r < nc_n; j_r += NR)
            {
              off_j = j_c + j_r;
              n = std_ext::min(NR, static_cast<dim_t>(nc_n - j_r));
              csc = C->col_stride_in_block(off_j / NR);

              for (size_t i_r = 0; i_r < mc_m_real; i_r += MR)
              {
                m = std_ext::min(MR, static_cast<dim_t>(mc_m_real - i_r));
                off_i = i_c + (i_r / 2);
                rsc = C->row_stride_in_block(off_i / MR);

                ctx->kernel(m, n, k, ctx->alpha, a_packed, b_packed, ctx->beta, c_result, 1, m, nullptr, ctx->cntx);

                unpack_1m_c(C, c_result, off_i, off_j, m, n, rsc, csc);

                a_packed += MR * k;
              }
              b_packed += k * NR;

              a_packed = a_packed_base;
            }
            b_packed = b_packed_base;
          }
        }
      }

      free(workspace);
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

      T* workspace = nullptr;
      T* a_packed = nullptr; // A in G^{MC x KC}
      T* a_packed_base = nullptr;

      T* b_packed = nullptr; // B in G^{KC x NC}
      T* b_packed_base = nullptr;

      T* c_result = nullptr; // C in G^{MC x NC}

      utils::alloc_aligned<T>(&workspace, MC * KC + KC * NC + MC * NC);

      a_packed = a_packed_base = workspace;
      b_packed = b_packed_base = workspace + MC * KC;
      c_result = workspace + MC * KC + KC * NC;

      size_t off_i, off_j;
      dim_t mc_m, nc_n, k, m, n;
      inc_t rsc = 1, csc;

      for (size_t j_c = 0; j_c < N; j_c += NC)
      {
        nc_n = std_ext::min(NC, static_cast<dim_t>(N - j_c));

        for (size_t p_c = 0; p_c < K; p_c += KC)
        {
          k = std_ext::min(KC, static_cast<dim_t>(K - p_c));
          pack_b(B, b_packed, p_c, j_c, k, nc_n, NR, KP);

          for (size_t i_c = 0; i_c < M; i_c += MC)
          {
            mc_m = std_ext::min(MC, static_cast<dim_t>(M - i_c));
            pack_a(A, a_packed, i_c, p_c, mc_m, k, MR, KP);

            for (size_t j_r = 0; j_r < nc_n; j_r += NR)
            {
              n = std_ext::min(NR, static_cast<dim_t>(nc_n - j_r));
              off_j = j_c + j_r;
              csc = C->col_stride_in_block(off_j / NR);

              for (size_t i_r = 0; i_r < mc_m; i_r += MR)
              {
                m = std_ext::min(MR, static_cast<dim_t>(mc_m - i_r));
                off_i = i_c + i_r;
                rsc = C->row_stride_in_block(off_i / MR);

                if (rsc > 0 && csc > 0)
                {
                  ctx->kernel(m, n, k,
                    ctx->alpha,
                    a_packed,
                    b_packed,
                    ctx->beta,
                    C->pointer_at_loc(off_i, off_j), rsc, csc,
                    nullptr,
                    ctx->cntx);
                }
                else {
                  ctx->kernel(m, n, k,
                    ctx->alpha,
                    a_packed,
                    b_packed,
                    ctx->beta,
                    c_result, 1, m,
                    nullptr,
                    ctx->cntx);

                  unpack_c_scat(C, c_result, off_i, off_j, m, n);
                }

                a_packed += MR * k;
              }
              b_packed += k * NR;

              a_packed = a_packed_base;
            }
            b_packed = b_packed_base;
          }
        }
      }

      free(workspace);
    }
  }
};
