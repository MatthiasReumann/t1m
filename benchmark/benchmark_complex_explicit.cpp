#include <iostream>
#include <chrono>
#include <numeric>
#include <complex>
#include <tuple>
#include "t1m/t1m.hpp"

constexpr int N = 10;

int main()
{
  std::complex<double>* A = nullptr, * B = nullptr, * C = nullptr;

  const std::vector<std::tuple<std::string, std::string, std::string>> contractions = {
        {"abcde", "efbad", "cf"}, /*a:48;c:24;b:32;e:48;d:32;f:32;*/
        {"abcde", "efcad", "bf"}, /* a:48;c:32;b:24;e:48;d:32;f:32;*/
        {"abcd", "dbea", "ec"}, /*a:72;c:24;b:72;e:72;d:72;*/
        {"abcde", "ecbfa", "fd"} /*a:48;c:32;b:32;e:48;d:24;f:48;*/,
        {"abcd", "deca", "be"} /*a:72;c:72;b:24;e:72;d:72;*/,
        {"abc", "bda", "dc"}, /*a:312;c:24;b:312;d:312;*/
        {"abcd", "ebad", "ce"}, /*a:72;c:24;b:72;e:72;d:72;*/

        {"abcdef", "dega", "gfbc"}, /*a:24;c:16;b:16;e:16;d:24;g:24;f:16;*/
        {"abcdef", "dfgb", "geac"}, /*a:24;c:16;b:16;e:16;d:24;g:24;f:16;*/
        {"abcdef", "degb", "gfac"}, /*a:24;c:16;b:16;e:16;d:24;g:24;f:16;*/
        {"abcdef", "degc", "gfab"}, /*a:24;c:16;b:16;e:16;d:24;g:24;f:16;*/
        
        {"abc", "dca", "bd"}, /* a:312;c:296;b:24;d:312; */
        
        {"abcd", "ea", "ebcd"}, /*a:72;c:72;b:72;e:72;d:72;*/
        {"abcd", "eb", "aecd"}, /*a:72;c:72;b:72;e:72;d:72;*/
        {"abcd", "ec", "abed"}, /*a:72;c:72;b:72;e:72;d:72;*/
        {"abc", "adec", "ebd"}, /*a:72;c:72;b:72;e:72;d:72;*/
        
        {"ab", "cad", "dcb"}, /*a:312;c:312;b:296;d:312;*/
        {"ab", "acd", "dbc"}, /*a:312;c:296;b:296;d:312;*/
        {"abc", "acd", "db"}, /*a:312;c:296;b:312;d:296;*/
        {"abc", "adc", "bd"}, /*a:312;c:296;b:312;d:296;*/
        
        {"ab", "ac", "cb"}, /*a:5136;c:5136;b:5120;*/

        {"abcd", "aebf", "fdec"}, /*a:72;c:72;b:72;e:72;d:72;f:72;*/
        {"abcd", "eafd", "fbec"}, /*a:72;c:72;b:72;e:72;d:72;f:72;*/
        {"abcd", "aebf", "dfce"}, /*a:72;c:72;b:72;e:72;d:72;f:72;*/
  };

  const std::vector<std::vector<std::vector<size_t>>> sizes = {
    {{24, 16, 12, 16, 24}, {24, 16, 16, 24, 16}, {12, 16}},
    {{24, 12, 16, 16, 24}, {24, 16, 16, 24, 16}, {12, 16}},
    {{36, 36, 12, 36}, {36, 36, 36, 36}, {36, 12}},
    {{24, 16, 16, 12, 24}, {24, 16, 16, 24, 24}, {24, 12}},
    {{36, 12, 36, 36}, {36, 36, 36, 36}, {12, 36}},
    {{156, 156, 12}, {156, 156, 156}, {156, 12}},
    {{36, 36, 12, 36}, {36, 36, 36, 36}, {12, 36}},

    {{12, 8, 8, 12, 8, 8}, {12, 8, 12, 12}, {12, 8, 8, 8}},
    {{12, 8, 8, 12, 8, 8}, {12, 8, 12, 8}, {12, 8, 12, 8}},
    {{12, 8, 8, 12, 8, 8}, {12, 8, 12, 8}, {12, 8, 12, 8}},
    {{12, 8, 8, 12, 8, 8}, {12, 8, 12, 8}, {12, 8, 12, 8}},

    {{156, 12, 148}, {156, 148, 156}, {12, 156}},

    {{36, 36, 36, 36}, {36, 36}, {36, 36, 36, 36}},
    {{36, 36, 36, 36}, {36, 36}, {36, 36, 36, 36}},
    {{36, 36, 36, 36}, {36, 36}, {36, 36, 36, 36}},
    {{36, 36, 36}, {36, 36, 36, 36}, {36, 36, 36}},

    {{156, 148}, {156, 156, 156}, {156, 156, 148}},
    {{156, 148}, {156, 148, 156}, {156, 148, 148}},
    {{156, 156, 148}, {156, 148, 148}, {148, 156}},
    {{156, 156, 148}, {156, 148, 148}, {156, 148}},

    {{2568, 2560}, {2568, 2568}, {2568, 2560}},

    {{36, 36, 36, 36}, {36, 36, 36, 36}, {36, 36, 36, 36}},
    {{36, 36, 36, 36}, {36, 36, 36, 36}, {36, 36, 36, 36}},
    {{36, 36, 36, 36}, {36, 36, 36, 36}, {36, 36, 36, 36}},
  };

  assert(sizes.size() == contractions.size());

  std::cout << "contraction,min" << '\n';
  int i = 0;
  for (const auto &[c, a, b] : contractions)
  {
    std::cout << c << "-" << a << "-" << b << ",";
    
    size_t sizeA = 1L;
    size_t sizeB = 1L;
    size_t sizeC = 1L;

    auto lengthsC = sizes[i][0];
    auto lengthsA = sizes[i][1];
    auto lengthsB = sizes[i][2];

    for (const auto &l : lengthsA) sizeA *= l;
    for (const auto &l : lengthsB) sizeB *= l;
    for (const auto &l : lengthsC) sizeC *= l;

    t1m::utils::alloc_aligned(&A, sizeA);
    t1m::utils::alloc_aligned(&B, sizeB);
    t1m::utils::alloc_aligned(&C, sizeC);

    auto tensorA = t1m::Tensor<std::complex<double>>(lengthsA, A);
    auto tensorB = t1m::Tensor<std::complex<double>>(lengthsB, B);
    auto tensorC = t1m::Tensor<std::complex<double>>(lengthsC, C);

    std::vector<double> time(N);
    for (uint j = 0; j < N; j++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      t1m::contract(tensorA, a, tensorB, b, tensorC, c);
      auto t1 = std::chrono::high_resolution_clock::now();
      time[j] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    std::cout << *min_element(time.begin(), time.end()) << '\n';

    free(A);
    free(B);
    free(C);

    i++;
  }
}