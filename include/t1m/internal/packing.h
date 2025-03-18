#include <algorithm>
#include <memory>
#include <span>
#include "t1m/internal/scatter.h"
#include "t1m/tensor.h"

namespace t1m::packing {

template <typename T>
class base {
 public:
  base(std::size_t size, std::allocator<T> alloc)
      : buf(alloc.allocate(size)), size(size), alloc(alloc) {}
  ~base() noexcept { alloc.deallocate(buf, size); }

 protected:
  T* buf;
  std::size_t size;
  std::allocator<T> alloc;
};

template <typename T>
struct pack_slivers_a {
  /// @brief The number of rows in this block.
  const std::size_t nrows;
  /// @brief The number of columns in this block.
  const std::size_t ncols;
  /// @brief The row stride for this block.
  const std::size_t rs;
  /// @brief The column stride for this block.
  const std::size_t cs;
  /// @brief Slice of full rscat relevant for this block.
  const std::span<const std::size_t> rscat;
  /// @brief Slice of full cscat relevant for this block.
  const std::span<const std::size_t> cscat;

  /// @brief
  /// @note  If row and column stride are constant, pack using constant stride
  ///        and continous access. Otherwise, pack using scatter layout.
  void operator()(const T* src, T* dest) noexcept {
    if (has_constant_stride()) {
      const std::size_t offset = (rscat[0] + cscat[0]);
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          dest[k + l * nrows] = src[k * rs + l * cs + offset];
        }
      }
    } else {
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          dest[k + l * nrows] = src[rscat[k] + cscat[l]];
        }
      }
    }
  }

 private:
  bool has_constant_stride() noexcept { return rs > 0 && cs > 0; }
};

template <typename T>
class lhs : public base<T> {
 public:
  lhs(std::size_t size, std::allocator<T> alloc = std::allocator<T>{})
      : base<T>(size, alloc) {}

  T* data() const noexcept { return this->buf; }

  /// @brief Pack `M` x `K` block into slivers.
  template <std::size_t ndim>
  void pack(const scatter::block_layout<T, ndim>& layout, const T* src,
            const std::size_t M, const std::size_t K) {

    const std::size_t nrowblocks = M / layout.row_size;
    const std::size_t remainder = M % layout.row_size;

    const std::vector<std::size_t> rows = layout.rows();
    const std::vector<std::size_t> cols = layout.cols();

    for (std::size_t i = 0; i < nrowblocks; ++i) {
      const std::size_t offset = i * K * layout.row_size;
      pack_slivers_a<T>{
          layout.row_size,
          K,
          layout.block_rscat[i],
          layout.block_cscat[i],
          std::span{rows}.subspan(i * layout.row_size, layout.row_size),
          std::span{cols}}(src, data() + offset);
    }

    if (remainder > 0) {
      const std::size_t offset = nrowblocks * K * layout.row_size;
      pack_slivers_a<T>{remainder,
                        K,
                        layout.block_rscat[i],
                        layout.block_cscat[i],
                        std::span{rows}.subspan(nrowblocks * layout.row_size,
                                                layout.row_size),
                        std::span{cols}}(src, data() + offset);
    }
  }
};

template <typename T>
struct rhs : public base<T> {
  rhs(std::size_t size, std::allocator<T> alloc = std::allocator<T>{})
      : base<T>(size, alloc) {}
  void pack(const T* orig) {}
};
};  // namespace t1m::packing