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

  void pack(const scatter::matrix_view& matrix, const T* src,
            const std::size_t M, const std::size_t K) {
    const std::size_t rbn = M / matrix.br;
    const std::size_t cbn = K / matrix.bc;

    const std::size_t rremainder = M % matrix.br;

    //       K
    //   ┌───┬───┐
    //   │   │   │
    // M ├───┼───┤ "block"
    //   │   │   │
    //   └───┴───┘

    for (std::size_t rb = 0; rb < rbn; ++rb) {
      const std::size_t roffset = cbn * rb * matrix.block_nelems();

      //        K
      //    ┌───┬───┐
      // br │   │   │ "sliver"
      //    └───┴───┘

      for (std::size_t cb = 0; cb < cbn; ++cb) {
        const std::size_t coffset = cb * matrix.block_nelems();

        //      bc
        //    ┌────┐
        // br │    │  "cell"
        //    └────┘

        const auto cell = matrix.subview(rb * matrix.br, cb * matrix.bc,
                                         matrix.br, matrix.bc);
        std::println("{} {} | {} {}", cell.rs, cell.cs, cell.rbs, cell.cbs);
        pack_cell(cell, src, data() + roffset + coffset);
      }
    }

    // if (remainder > 0) {
    //   const std::size_t offset = nblocks * K * layout.row_size;
    //   pack_slivers_a<T>{
    //       remainder,
    //       K,
    //       layout.block_rscat[nblocks],
    //       layout.block_cscat[nblocks],
    //       std::span{rows}.subspan(nblocks * layout.row_size, layout.row_size),
    //       std::span{cols}}(src, data() + offset);
    // }
  }
};
};  // namespace t1m::packing