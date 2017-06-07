#ifndef TENSOR_CONTRACT_HH
#define TENSOR_CONTRACT_HH

#include<cassert>

namespace tensors{

//! Expression template for generic tensor contractions
template<size_t i1, size_t i2, typename E1, typename E2>
class tensor_contraction_t : public tensor_expression_t<tensor_contraction_t<i1,i2,E1, E2> > {
    //! references to both tensors
    E1 const& _u;
    E2 const& _v;

  public:
    //! Usual definitions, see tensor definition
    using data_t = typename E1::data_t;
    using frame_t = typename E1::frame_t;
    static constexpr size_t ndim = E1::ndim;
    static constexpr size_t rank = E1::rank + E2::rank -2;

    static inline constexpr decltype(auto) get_index_t(){
      // create index tuples for both tensors
      constexpr auto E1_indices = typename E1::index_t();
      constexpr auto E2_indices = typename E2::index_t();

      constexpr size_t E1_size = E1::rank;
      constexpr size_t E2_size = E2::rank;

      // create subtuples
      constexpr auto E1_p1 = get_subtuple<E1_size*(i1<1),(i1-1)*(i1>1)>(E1_indices);
      constexpr auto E1_p2 = get_subtuple<i1+1,E1_size-1>(E1_indices);

      constexpr auto E2_p1 = get_subtuple<E2_size*(i2<1),(i2-1)*(i2>1)>(E2_indices);
      constexpr auto E2_p2 = get_subtuple<i2+1,E2_size-1>(E2_indices);

      return std::tuple_cat(E1_p1,E1_p2,E2_p1,E2_p2);
    }

    using index_t = decltype(get_index_t());

    using this_tensor_t = general_tensor_t<data_t,frame_t,rank,index_t,ndim>;

    tensor_contraction_t(E1 const& u, E2 const& v) : _u(u), _v(v) {

      static_assert(std::is_same<typename E1::frame_t, typename E2::frame_t>::value,
                    "Frame types don't match!");

      static_assert(E1::ndim == E2::ndim,
		                "Dimensions don't match!");

      static_assert(std::is_same<typename E1::data_t, typename E2::data_t>::value,
		                "Data types don't match!");

      static_assert(std::is_same<
                      typename std::conditional<
                        std::is_same<
					                typename std::tuple_element<i1,typename E1::index_t>::type,
					                lower_t
					              >::value,
					              lower_t,
					              upper_t
					            >::type,
					            typename std::conditional<
					              std::is_same<
					                typename std::tuple_element<i2,typename E2::index_t>::type,
					                upper_t
					              >::value,
					              lower_t,
					              upper_t
					            >::type
                    >::value,
                    "Ranks don't match! Can only contract covariant with contravariant indices!");

      static_assert(rank == std::tuple_size<index_t>::value ,
                    "Index tuple size != rank, this should not happen");
    };

    // CHECK: is this necessary? Imagine needing only one component of a contraction.
    //        point-wise evaluation is also defined for the other expressions
    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline decltype(auto) operator[](size_t i) const = delete;

    //! Sum of contracted components for a specific index computed by a template recursion
    template<int N, int stride1, int stride2>
    struct recursive_contract {
      template<typename A, typename B>
      static inline decltype(auto) contract(A const & _u, B const & _v) {
        return recursive_contract<(N-1),stride1,stride2>::contract(_u,_v)
             + _u.template evaluate<stride1 + N*utilities::static_pow<ndim,i1>::value>()
             * _v.template evaluate<stride2 + N*utilities::static_pow<ndim,i2>::value>();
      };
    };
    template<int stride1, int stride2>
    struct recursive_contract<0,stride1,stride2> {
      template<typename A, typename B>
      static inline decltype(auto) contract(A const & _u, B const & _v){
                    return _u.template evaluate<stride1>()*_v.template evaluate<stride2>() ;
      };
    };


    template<size_t c_index>
    inline decltype(auto) evaluate() const {
      // Uncompress index of resulting tensor
      constexpr auto index_r = uncompress_index<ndim,rank,c_index>();
      constexpr size_t index_size = std::tuple_size<decltype(index_r)>::value;

      // Need to restore the indices of the two tensors involved
      constexpr auto t1 = std::make_tuple(static_cast<size_t>(0));
      constexpr auto t2 = std::make_tuple(static_cast<size_t>(0));

      constexpr auto E1_size = std::tuple_size<typename E1::index_t>::value;
      constexpr auto E2_size = std::tuple_size<typename E2::index_t>::value;

      // Create subtuples
      constexpr auto E1_p1 = get_subtuple<index_size*(i1<1),(i1-1)*(i1>1)>(index_r);
      constexpr auto E1_p2 = get_subtuple<i1 +(index_size)*(E1_size<2) ,(E1_size-2)*(E1_size>2) >(index_r);

      constexpr auto E2_p1 = get_subtuple<E1_size-1 +index_size*(i2 + E1_size -1 <1) ,(E1_size-1 + i2-1)*(E1_size-1+i2>1) >(index_r);
      constexpr auto E2_p2 = get_subtuple<E1_size-1 +i2, rank - 1 >(index_r);

      // Concatenate tuples to a single tuple
      constexpr auto index_1 = std::tuple_cat(E1_p1,t1,E1_p2);
      constexpr auto index_2 = std::tuple_cat(E2_p1,t2,E2_p2);

      // Get compressed index from index tuples of contracted tensors
      constexpr size_t stride_1 =E1::this_tensor_t::compressed_index(index_1);
      constexpr size_t stride_2 =E2::this_tensor_t::compressed_index(index_2);


      static_assert(stride_1>=0,
                    "contraction: stride is less than zero, this shouldn't happen");
      static_assert(stride_2>=0,
                    "contraction: stride is less than zero, this shouldn't happen");

      // Compute sum over contracted index for index_r'th component by template recursion
      return recursive_contract<ndim-1,stride_1,stride_2>::contract(_u,_v);
    }
};

//! Contraction "operator" for two tensor expressions
template<size_t i1, size_t i2, typename E1, typename E2>
tensor_contraction_t<i1,i2,E1,E2> const
inline contract(E1 const &u, E2 const &v) {
  return tensor_contraction_t<i1,i2,E1,E2>(u,v);
}

}
#endif
