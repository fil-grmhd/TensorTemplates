#ifndef TENSOR_CONTRACT_HH
#define TENSOR_CONTRACT_HH

#include<cassert>

namespace tensors{


template<size_t i1, size_t i2, typename E1, typename E2>
class tensor_contraction_t : public tensor_expression_t<tensor_contraction_t<i1,i2,E1, E2> > {
   E1 const& _u;
   E2 const& _v;
    
public:

    using data_t = typename E1::data_t;
    using frame_t = typename E1::frame_t;
    static constexpr size_t ndim = E1::ndim;
    static constexpr size_t rank = E1::rank + E2::rank -2;


    static inline constexpr decltype(auto) get_index_t(){

      using E1_t = typename std::tuple_element<i1,typename E1::index_t>::type;
      using E2_t = typename std::tuple_element<i2,typename E2::index_t>::type;

      constexpr auto E1_it = typename E1::index_t(); 
      constexpr auto E2_it = typename E2::index_t(); 

      constexpr auto E1_size = std::tuple_size<typename E1::index_t>::value;
      constexpr auto E2_size = std::tuple_size<typename E2::index_t>::value;

      constexpr auto E1_p1 = get_subtuple<0,i1-1>(E1_it);
      constexpr auto E1_p2 = get_subtuple<i1+1,E1_size-1>(E1_it);

      constexpr auto E2_p1 = get_subtuple<E1_size-1,i2-1>(E2_it);
      constexpr auto E2_p2 = get_subtuple<i2+1,E2_size-1>(E2_it);

      constexpr auto index_1 = std::tuple_cat(E1_p1,E1_p2);
      constexpr auto index_2 = std::tuple_cat(E2_p2,E2_p2);

      return std::tuple_cat(index_1,index_2);
    
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
					lower_t>::value,
					lower_t,
					upper_t
					>::type,
					
					typename std::conditional<
					std::is_same< 
					typename std::tuple_element<i2,typename E2::index_t>::type,
					upper_t>::value,
					lower_t,
					upper_t
					>::type

		                        >::value,
		"Ranks don't match! Can only contract covariant with contravariant indices!");
    };
    
    inline decltype(auto) operator[](size_t i) const { 
      assert(!"Do not acces the tensor expression via the [] operator!");
      return 0.; };

    template<int N=ndim-1, size_t stride1, size_t stride2>
    inline decltype(auto) recursive_contract(){
      using namespace utilities;

      return (N==0) ? _u.template evaluate<stride1>()*_v.template evaluate<stride2>() 
	            : recursive_contract<(N-1)*(N>0),stride1,stride2>() + 
		      _u.template evaluate<stride1 + (N-1)*static_pow<ndim,i1>::value>()
		      *_v.template evaluate<stride2 + (N-1)*static_pow<ndim,i2>::value>();
    }


    template<size_t cindex>
    inline decltype(auto) evaluate() const { 

      constexpr auto index_r = uncompress_index<ndim,rank,cindex>();

      constexpr size_t index_size = std::tuple_size<decltype(index_r)>::value;

      //Need to restore the indices of the two tensors involved
      constexpr auto t1 = std::make_tuple(static_cast<size_t>(0));
      constexpr auto t2 = std::make_tuple(static_cast<size_t>(0));

      constexpr auto E1_size = std::tuple_size<typename E1::index_t>::value;
      constexpr auto E2_size = std::tuple_size<typename E2::index_t>::value;
  
      constexpr auto E1_p1 = get_subtuple<0,i1-1>(index_r);
      constexpr auto E1_p2 = get_subtuple<i1,E1_size-1>(index_r);

      constexpr auto E2_p1 = get_subtuple<E1_size-1,i2-1>(index_r);
      constexpr auto E2_p2 = get_subtuple<i2,E2_size-1>(index_r);

      constexpr auto index_1 = std::tuple_cat(E1_p1,t1,E1_p2);
      constexpr auto index_2 = std::tuple_cat(E2_p2,t2,E2_p2);

      constexpr size_t stride_1 =E1::this_tensor_t::compressed_index(index_1);
      constexpr size_t stride_2 =E2::this_tensor_t::compressed_index(index_2);

      return recursive_contract<ndim-1,stride_1,stride_2>(); 
     
  }
  
};

template<size_t i1, size_t i2, typename E1, typename E2>
tensor_contraction_t<i1,i2,E1,E2> const
inline contract( E1 const &u, E2 const &v){
  return tensor_contraction_t<i1,i2,E1,E2>(u,v);
}

}
#endif
