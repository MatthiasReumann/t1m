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
class lhs : public base<T> {
 public:
  lhs(std::size_t size, std::allocator<T> alloc = std::allocator<T>{})
      : base<T>(size, alloc) {}

  T* data() const noexcept { return this->buf; }

  void pack_cell(const scatter::matrix_view& cell, const T* src, T* dest) {
    const std::size_t nrows = cell.nrows();
    const std::size_t ncols = cell.ncols();

    const std::size_t rs = cell.rbs[0];
    const std::size_t cs = cell.cbs[0];

    if (rs > 0 && cs > 0) {
      const std::size_t offset = (cell.rs[0] + cell.cs[0]);
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          dest[k + l * nrows] = src[k * rs + l * cs + offset];
        }
      }
    } else {
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          dest[k + l * nrows] = src[cell.rs[k] + cell.cs[l]];
        }
      }
    }
  }

  void pack_block(const scatter::matrix_view& matrix, const T* src,
                  const std::size_t M, const std::size_t K) {

    //       K
    //   ┌───┬───┐
    //   │   │   │
    // M ├───┼───┤ "block"
    //   │   │   │
    //   └───┴───┘

    std::size_t offset = 0;
    for (std::size_t r = 0; r < M; r += matrix.br) {

      //        K
      //    ┌───┬───┐
      // br │   │   │ "sliver"
      //    └───┴───┘

      for (std::size_t c = 0; c < K; c += matrix.bc) {

        //      bc
        //    ┌────┐
        // br │    │  "cell"
        //    └────┘

        const auto cell = matrix.subview(r, c, std::min(matrix.br, M - r),
                                         std::min(matrix.bc, K - c));
        pack_cell(cell, src, data() + offset);
        offset += cell.block_nelems();
      }
    }
  }
};
};  // namespace t1m::packing