/*
 * =====================================================================================
 *
 *       Filename:  tensor_index.hh
 *
 *    Description:  Index handling for tensor contractions
 *
 *        Version:  1.0
 *        Created:  04/06/2017 14:52:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), emost@itp.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#ifndef TENSOR_INDEX_HH
#define TENSOR_INDEX_HH

#include <tuple>

namespace tensors {

//! Computes the compressed index associated with the given indices
/*!
 *  The data is stored in a column major format
 *  This is not as efficient as the compressed_index_t below
 */
template<size_t ndim>
constexpr inline size_t compress_indices(size_t const a) {
  return a;
}
template<size_t ndim, typename... indices_t>
constexpr inline size_t compress_indices(size_t const a, indices_t... indices) {
  return a + compress_indices<ndim>(indices...)*ndim;
}

//! Computes the compressed index given template parameter indices (in column-major format)
template <size_t ndim, size_t a, size_t... indices>
struct compressed_index_t {
  static constexpr size_t value =
      compressed_index_t<ndim, indices...>::value * ndim + a;
};
// termination definition of recursion
template <size_t ndim, size_t a>
struct compressed_index_t<ndim, a> {
  static constexpr size_t value = a;
};

//! Computes the compressed index of a tuple of indices
template <size_t ndim, typename tuple_t,
          size_t N = std::tuple_size<tuple_t>::value - 1>
static inline constexpr size_t compressed_index_tuple(tuple_t t) {
  return (N == 0)
             ? std::get<std::tuple_size<tuple_t>::value - 1>(t)
             : compressed_index_tuple<ndim, tuple_t, (N - 1) * (N > 0)>(t) *
                       ndim +
                   std::get<(std::tuple_size<tuple_t>::value - 1 - N) *
                            (N > 0)>(t);
}

//! Uncompresses single index of position index_pos from compressed index
//! c_index
template <size_t ndim, size_t index_pos, size_t c_index>
struct uncompress_index_t {
  static constexpr size_t value =
      static_cast<size_t>(c_index /
                          utilities::static_pow<ndim, index_pos>::value) %
      ndim;
};

// Count number of free indices, defined by an index < 0
template <int i0, int... Indices>
struct count_free_indices {
  static constexpr size_t value =
      (i0 < 0) + count_free_indices<Indices...>::value;
};
template <int i0>
struct count_free_indices<i0> {
  static constexpr size_t value = (i0 < 0);
};

// Pick shift from first free index
template <int i0, int... Indices>
struct free_index_shift {
  static constexpr int value = (i0 < 0)*(-(i0 + 1)) + (i0 >= 0)*free_index_shift<Indices...>::value;
};
template <int i0>
struct free_index_shift<i0> {
  static constexpr int value = (i0 < 0)*(-(i0 + 1));
};

// transforms the compressed index of a slice
// to a compressed index of the underlying expression
template <typename E, typename E_sliced, size_t c_index, size_t N_slice, int ind0, int... Indices>
struct compute_unsliced_cindex {
  static constexpr size_t shift = static_cast<size_t>(-ind0) - 1;
  static constexpr size_t value =
      (ind0 >= 0) * ind0
                  * utilities::static_pow<
                      E::property_t::ndim,
                      E::property_t::rank - sizeof...(Indices) - 1
                    >::value
     + (ind0 < 0) * (uncompress_index_t<
                       E_sliced::property_t::ndim,
                       N_slice,
                       c_index
                     >::value + shift)
                  * utilities::static_pow<
                      E::property_t::ndim,
                      E::property_t::rank - sizeof...(Indices) - 1
                    >::value
      // Recursion
   + compute_unsliced_cindex<E, E_sliced, c_index, N_slice + (ind0 < 0), Indices...>::value;
};
template <typename E, typename E_sliced, size_t c_index, size_t N_slice, int ind0>
struct compute_unsliced_cindex<E,E_sliced,c_index,N_slice,ind0> {
  static constexpr size_t shift = static_cast<size_t>(-ind0) - 1;
  static constexpr size_t value =
      (ind0 >= 0) * ind0
                  * utilities::static_pow<
                      E::property_t::ndim,
                      E::property_t::rank - 1
                    >::value
     + (ind0 < 0) * (uncompress_index_t<
                       E_sliced::property_t::ndim,
                       N_slice,
                       c_index
                     >::value + shift)
                  * utilities::static_pow<
                      E::property_t::ndim,
                      E::property_t::rank - 1
                    >::value;
};


// The following stuff doesn't work with intel compiler 17.0
// https://software.intel.com/en-us/forums/intel-c-compiler/topic/737195

template <size_t ndim, size_t c_index, std::size_t... I>
constexpr decltype(auto) uncompress_index_impl(std::index_sequence<I...>) {
  // creates a tuple of uncompressed indices for index 0,...,rank-1
  return std::make_tuple((uncompress_index_t<ndim, I, c_index>::value)...);
};

//! Creates a tuple of rank indices from a given compressed index c_index
template <size_t ndim, size_t rank, size_t c_index,
          typename Indices = std::make_index_sequence<rank>>
constexpr decltype(auto) uncompress_index() {
  // passes index sequence, i.e. uncompress_index_impl<ndim,
  // c_index>(0,...,rank-1)
  return uncompress_index_impl<ndim, c_index>(Indices{});
};

template <size_t offset, typename tuple_t, size_t... I>
constexpr decltype(auto) get_subtuple_impl(const tuple_t &t,
                                           std::index_sequence<I...>) {
  return std::make_tuple((std::get<offset + I>(t))...);
};

template <size_t begin, size_t end, typename tuple_t,
          typename Indices = std::make_index_sequence<
              (std::min(std::tuple_size<tuple_t>::value - 1, end) + 1 > begin) *
              (std::min(std::tuple_size<tuple_t>::value - 1, end) - begin + 1)>>
constexpr decltype(auto) get_subtuple(const tuple_t &t) {
  return get_subtuple_impl<begin>(t, Indices{});
};
};

#endif
