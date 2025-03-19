#pragma once
#include <array>
#include <cstddef>
#include "t1m/internal/utils.h"
#include "t1m/tensor.h"

namespace t1m::scatter {

struct matrix_view {
  std::span<const std::size_t> rs;
  std::span<const std::size_t> cs;

  const std::size_t br;
  std::span<const std::size_t> rbs;

  const std::size_t bc;
  std::span<const std::size_t> cbs;

  constexpr matrix_view subview(std::size_t ri, std::size_t ci, std::size_t nrows,
                      std::size_t ncols) const noexcept {
    const std::size_t rfrstblck = ri / br;
    const std::size_t cfrstblck = ci / bc;
    const std::size_t rnblcks = (nrows + br - 1) / br;  // ceil(nrows / br)
    const std::size_t cnblcks = (ncols + bc - 1) / bc;  // ceil(ncols / bc)
    return {rs.subspan(ri, nrows),
            cs.subspan(ci, ncols),
            br,
            rbs.subspan(rfrstblck, rnblcks),
            bc,
            cbs.subspan(cfrstblck, cnblcks)};
  }

  constexpr std::size_t nrows() const noexcept { return rs.size(); }
  constexpr std::size_t ncols() const noexcept { return cs.size(); }
  constexpr std::size_t block_nelems() const noexcept { return nrows() * ncols(); }
};

struct block_layout {
  template <const std::size_t ndim>
  block_layout(const std::array<std::size_t, ndim> dimensions,
               const std::array<std::size_t, ndim> strides,
               const std::vector<std::size_t>& row_indices,
               const std::vector<std::size_t>& col_indices,
               const std::size_t br, const std::size_t bc)
      : br(br), bc(bc) {
    rs = utils::scatter<ndim>{}(row_indices, dimensions, strides);
    cs = utils::scatter<ndim>{}(col_indices, dimensions, strides);
    rbs = utils::block_scatter{br}(rs);
    cbs = utils::block_scatter{bc}(cs);
  }

  std::vector<std::size_t> rs;
  std::vector<std::size_t> cs;

  std::size_t br;
  std::vector<std::size_t> rbs;
  std::size_t bc;
  std::vector<std::size_t> cbs;
};

// template <typename T, const std::size_t ndim>
// class matrix_view {
//  public:
//   matrix_view(const tensor<T, ndim>& t,
//               const std::vector<std::size_t>& row_indices, const std::size_t br,
//               const std::vector<std::size_t>& col_indices, const std::size_t bc)
//       : rscat(utils::scatter<ndim>{}(row_indices, t.dimensions, t.strides())),
//         cscat(utils::scatter<ndim>{}(col_indices, t.dimensions, t.strides())),
//         br(br),
//         bc(bc),
//         block_rscat(utils::block_scatter{br}(rscat)),
//         block_cscat(utils::block_scatter{bc}(cscat)) {}

//   matrix_subview subview(std::size_t ri, std::size_t ci, std::size_t nrows,
//                          std::size_t ncols) const noexcept {
//     return {std::span{rscat}.subspan(ri, nrows),
//             std::span{cscat}.subspan(ci, ncols),
//             br,
//             std::span{block_rscat},
//             bc,
//             std::span{block_cscat}};
//   }

//   std::size_t nrows() const noexcept { return rscat.size(); }
//   std::size_t ncols() const noexcept { return cscat.size(); }

//  private:
//   std::vector<std::size_t> rscat;
//   std::vector<std::size_t> cscat;

//   std::size_t br;
//   std::size_t bc;
//   std::vector<std::size_t> block_rscat;
//   std::vector<std::size_t> block_cscat;
// };
};  // namespace t1m::scatter