#pragma once

namespace t1m
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