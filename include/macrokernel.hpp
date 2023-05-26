#pragma once

/*
A = G^{mc x kc}, B = G^{kc x nc}, C = G^{mc x nc}
C = A . B
*/
template <int mc, int nc, int kc>
void macrokernel_simple(float *A, float *B, float *C)
{

  for (int i = 0; i < mc; i++)
  {
    for (int j = 0; j < nc; j++)
    {
      float c_ij = 0.;
      for (int p = 0; p < kc; p++)
      {
        const float a = A[i + p * mc];
        const float b = B[p + j * kc];
        c_ij += a * b;
      }

      C[i + j * mc] = c_ij;
    }
  }
}