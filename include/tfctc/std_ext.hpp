#pragma once

namespace tfctc
{
  namespace std_ext
  {
    template <typename T>
    inline T min(T a, T b)
    {
      if (a <= b)
      {
        return a;
      }
      return b;
    }
  };
};