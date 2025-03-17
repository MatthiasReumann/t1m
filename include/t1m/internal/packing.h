#include <memory>
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
class lhs : public base<T> {
 public:
  lhs(std::size_t size, std::allocator<T> alloc = std::allocator<T>{})
      : base<T>(size, alloc) {}

  T* data() const noexcept { return this->buf; }

  /// @brief Pack `M` x `K` block into slivers.
  template <std::size_t ndim>
  void pack(const scatter::block_layout<T, ndim>& layout, const T* src,
            const std::size_t M, const std::size_t K) {

    const std::size_t nrowblocks = (M / layout.row_size);
    const std::size_t ncolblocks = (K / layout.col_size);

    for (std::size_t i = 0; i < nrowblocks; ++i) {
      const std::size_t rs = layout.block_rscat[i];

      for (std::size_t j = 0; j < ncolblocks; ++j) {
        const std::size_t cs = layout.block_cscat[i];

        // If row and column stride are constant,
        // pack using constant stride and continous access.
        // Otherwise, pack using scatter layout.
        //
        // The `data` buffer offset is the amount of previously packed elements.
        //
        // Use the scatter layout to apply the correct offset to the `src` buffer.
        // - For the continous access case, simply add it.
        // - For the access via the scatter layout, add it to the respecive row and
        //   column indices.

        const std::size_t offset = layout.nelem() * (j + i * ncolblocks);
        if (rs > 0 && cs > 0) {
          for (std::size_t k = 0; k < layout.row_size; ++k) {
            for (std::size_t l = 0; l < layout.col_size; ++l) {
              data()[k + l * layout.row_size + offset] =
                  src[layout(layout.row_size * i, layout.col_size * j) +
                      k * rs + l * cs];
            }
          }
        } else {
          for (std::size_t k = 0; k < layout.row_size; ++k) {
            for (std::size_t l = 0; l < layout.col_size; ++l) {
              data()[k + l * layout.row_size + offset] =
                  src[layout(k + layout.row_size * i, l + layout.col_size * j)];
            }
          }
        }
      }
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