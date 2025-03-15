// # pragma once 

// #include <complex>
// #include "block_scatter_matrix.hpp"
// #include "packs_1m.hpp"
// #include "blis.h"

// namespace t1m::utils
// {
//   template <typename U>
//   void pack_1m_a(BlockScatterMatrix<std::complex<U>>* A, U* buffer, int off_i, int off_j, dim_t M, dim_t K, const dim_t MR, const dim_t KP)
//   {
//     const size_t M_blocks = size_t(M / MR);
//     const size_t K_blocks = size_t(K / KP);

//     const size_t m1 = M % MR;
//     const size_t k1 = K % KP;

//     const size_t off_j_bak = off_j;
//     const size_t start_b_m = size_t(off_i / MR);
//     const size_t start_b_k = size_t(off_j / KP);

//     // next sliver offsets
//     const size_t offns_fullblocks = (K - off_j_bak) * 2 * MR;
//     const size_t offns = offns_fullblocks + 2 * k1 * MR;

//     const dim_t HALFMR = MR / 2;

//     size_t m, k, bm;
//     inc_t rsa, csa;

//     for (m = start_b_m; m < M_blocks; m++)
//     {
//       rsa = A->row_stride_in_block(m);

//       for (k = start_b_k; k < K_blocks; k++)
//       {
//         csa = A->col_stride_in_block(k);

//         if (rsa > 0 && csa > 0)
//         {
//           pack_1m_as_cont(A->pointer_at_loc(off_i, off_j), buffer, HALFMR, KP, MR, rsa, csa);
//           pack_1m_as_cont(A->pointer_at_loc(off_i + HALFMR, off_j), buffer + offns, HALFMR, KP, MR, rsa, csa);
//         }
//         else {
//           pack_1m_as_scat(A, buffer, HALFMR, KP, MR, off_i, off_j);
//           pack_1m_as_scat(A, buffer + offns, HALFMR, KP, MR, off_i + HALFMR, off_j);
//         }

//         buffer += 2 * MR * KP;
//         off_j += KP;
//       }

//       if (k1 > 0)
//       {
//         csa = A->col_stride_in_block(k);

//         if (rsa > 0 && csa > 0)
//         {
//           pack_1m_as_cont(A->pointer_at_loc(off_i, off_j), buffer, HALFMR, k1, MR, rsa, csa);
//           pack_1m_as_cont(A->pointer_at_loc(off_i + HALFMR, off_j), buffer + offns, HALFMR, k1, MR, rsa, csa);
//         }
//         else {
//           pack_1m_as_scat(A, buffer, HALFMR, k1, MR, off_i, off_j);
//           pack_1m_as_scat(A, buffer + offns, HALFMR, k1, MR, off_i + HALFMR, off_j);
//         }

//         buffer += 2 * MR * k1;
//       }

//       off_i += MR;
//       off_j = off_j_bak;

//       buffer += offns;
//     }

//     if (m1 > 0)
//     {
//       rsa = A->row_stride_in_block(m);

//       if (m1 > HALFMR)
//       {
//         for (k = start_b_k; k < K_blocks; k++)
//         {
//           csa = A->col_stride_in_block(k);

//           if (rsa > 0 && csa > 0)
//           {
//             pack_1m_as_cont(A->pointer_at_loc(off_i, off_j), buffer, HALFMR, KP, MR, rsa, csa);
//             pack_1m_as_cont(A->pointer_at_loc(off_i + HALFMR, off_j), buffer + offns, m1 - HALFMR, KP, MR, rsa, csa);
//           }
//           else {
//             pack_1m_as_scat(A, buffer, HALFMR, KP, MR, off_i, off_j);
//             pack_1m_as_scat(A, buffer, m1 - HALFMR, KP, MR, off_i + HALFMR, off_j);
//           }

//           buffer += 2 * MR * KP;
//           off_j += KP;
//         }

//         if (k1 > 0)
//         {
//           csa = A->col_stride_in_block(k);

//           if (rsa > 0 && csa > 0)
//           {
//             pack_1m_as_cont(A->pointer_at_loc(off_i, off_j), buffer, HALFMR, k1, MR, rsa, csa);
//             pack_1m_as_cont(A->pointer_at_loc(off_i + HALFMR, off_j), buffer + offns_fullblocks, m1 - HALFMR, k1, MR, rsa, csa);
//           }
//           else {
//             pack_1m_as_scat(A, buffer, HALFMR, k1, MR, off_i, off_j);
//             pack_1m_as_scat(A, buffer + offns_fullblocks, m1 - HALFMR, k1, MR, off_i + HALFMR, off_j);
//           }
//         }
//       }
//       else
//       {
//         for (k = start_b_k; k < K_blocks; k++)
//         {
//           csa = A->col_stride_in_block(k);

//           if (rsa > 0 && csa > 0)
//           {
//             pack_1m_as_cont(A->pointer_at_loc(off_i, off_j), buffer, m1, KP, MR, rsa, csa);
//           }
//           else {
//             pack_1m_as_scat(A, buffer, m1, KP, MR, off_i, off_j);
//           }

//           buffer += 2 * MR * KP;
//           off_j += KP;
//         }

//         if (k1 > 0)
//         {
//           csa = A->col_stride_in_block(k);

//           if (rsa > 0 && csa > 0)
//           {
//             pack_1m_as_cont(A->pointer_at_loc(off_i, off_j), buffer, m1, k1, MR, rsa, csa);
//           }
//           else {
//             pack_1m_as_scat(A, buffer, m1, k1, MR, off_i, off_j);
//           }
//         }
//       }
//     }
//   }

//   template <typename U>
//   void pack_1m_b(BlockScatterMatrix<std::complex<U>>* B, U* buffer, int off_i, int off_j, dim_t K, dim_t N, const dim_t NR, const dim_t KP)
//   {
//     const dim_t k1 = K % KP;
//     const dim_t n1 = N % NR;
//     const size_t off_i_bak = off_i;

//     const size_t start_b_k = size_t(off_i / KP);
//     const size_t start_b_n = size_t(off_j / NR);

//     size_t k, n;
//     inc_t rsb, csb;

//     for (n = start_b_n; n < size_t(N / NR); n++)
//     {
//       csb = B->col_stride_in_block(n);

//       for (k = start_b_k; k < size_t(K / KP); k++)
//       {
//         rsb = B->row_stride_in_block(k);

//         if (rsb > 0 && csb > 0)
//         {
//           pack_1m_bs_cont(B->pointer_at_loc(off_i, off_j), buffer, KP, NR, NR, rsb, csb);
//         }
//         else {
//           pack_1m_bs_scat(B, buffer, KP, NR, NR, off_i, off_j);
//         }

//         buffer += NR * (2 * KP);
//         off_i += KP;
//       }

//       if (k1 > 0)
//       {
//         rsb = B->row_stride_in_block(k);

//         if (rsb > 0 && csb > 0)
//         {
//           pack_1m_bs_cont(B->pointer_at_loc(off_i, off_j), buffer, k1, NR, NR, rsb, csb);
//         }
//         else {
//           pack_1m_bs_scat(B, buffer, k1, NR, NR, off_i, off_j);
//         }

//         buffer += NR * (2 * k1);
//       }

//       off_j += NR;
//       off_i = off_i_bak;
//     }

//     if (n1 > 0)
//     {
//       csb = B->col_stride_in_block(n);

//       for (k = start_b_k; k < size_t(K / KP); k++)
//       {
//         rsb = B->row_stride_in_block(k);

//         if (rsb > 0 && csb > 0)
//         {
//           pack_1m_bs_cont(B->pointer_at_loc(off_i, off_j), buffer, KP, n1, NR, rsb, csb);
//         }
//         else {
//           pack_1m_bs_scat(B, buffer, KP, n1, NR, off_i, off_j);
//         }

//         buffer += NR * (2 * KP);
//         off_i += KP;
//       }

//       if (k1 > 0)
//       {
//         rsb = B->row_stride_in_block(k);

//         if (rsb > 0 && csb > 0)
//         {
//           pack_1m_bs_cont(B->pointer_at_loc(off_i, off_j), buffer, k1, n1, NR, rsb, csb);
//         }
//         else {
//           pack_1m_bs_scat(B, buffer, k1, n1, NR, off_i, off_j);
//         }
//       }
//     }
//   }

//   template <typename U>
//   void unpack_1m_c(BlockScatterMatrix<std::complex<U>>* C, U* c_result, size_t off_i, size_t off_j, dim_t M, dim_t N, inc_t rs, inc_t cs)
//   {
//     if (rs > 0 && cs > 0) unpack_1m_c_cont(C->pointer_at_loc(off_i, off_j), c_result, M, N, rs, cs);
//     else unpack_1m_c_scat(C, c_result, off_i, off_j, M, N);
//   }
// };