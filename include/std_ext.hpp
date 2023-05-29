namespace std_ext
{
  template <typename T>
  T min(T a, T b)
  {
    if (a <= b)
    {
      return a;
    }
    return b;
  }
}