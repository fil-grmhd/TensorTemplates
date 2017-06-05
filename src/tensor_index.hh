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

namespace tensors{

// termination definition of recursion
template<size_t ndim, size_t a>
struct compress_index_t {
  static constexpr size_t value = a;
};

//! Computes the compressed index (in row-major format)
template<size_t ndim, size_t a, size_t... indices>
struct compressed_index_t {
  static constexpr size_t value = compress_index_t<ndim,indices...>::value * ndim + a;
};

//! Computes the compressed index of a tuple of indices
template<size_t ndim, typename tuple_t, size_t N =std::tuple_size<tuple_t>::value - 1>
static inline constexpr size_t compressed_index_tuple(tuple_t t) {
  return (N==0) ? std::get<std::tuple_size<tuple_t>::value-1 -N>(t)
                : compressed_index_tuple<ndim, tuple_t,(N-1)*(N>0)>(t) * ndim + std::get<(N-1)*(N>0)>(t) ;
}

//Obtain index from compressed index
template<size_t ndim, size_t rank, size_t c_index>
struct uncompress_index_t {
   static constexpr size_t value = (c_index/utilities::static_pow<ndim,rank>::value) % ndim;
};

template<size_t ndim, size_t c_index, std::size_t... I>
constexpr decltype(auto) uncompress_index_impl(std::index_sequence<I...>) {
     return std::make_tuple((uncompress_index_t<ndim,I,c_index>::value)...);
};

template<size_t ndim, size_t rank, size_t c_index, typename Indices= std::make_index_sequence<rank>>
constexpr decltype(auto) uncompress_index() {
   return uncompress_index_impl<ndim, c_index>(Indices{});
};

template<size_t offset, typename tuple_t, size_t... I>
constexpr decltype(auto) get_subtuple_impl(const tuple_t &t, std::index_sequence<I...>) {
  return std::make_tuple((std::get<offset + I>(t))...);
};

template<size_t begin, size_t end,
         typename tuple_t,
         typename Indices = std::make_index_sequence<(
                              std::min(std::tuple_size<tuple_t>::value-1,end)+1 > begin)
                              *(std::min(std::tuple_size<tuple_t>::value-1,end)-begin+1)
                            >
         >
constexpr decltype(auto) get_subtuple(const tuple_t &t) {
   return get_subtuple_impl<begin>(t, Indices{});
};

};

#endif
