compress_index<ndim, rank, c_index>();
    constexpr size_t index_size = std::tuple_size<decltype(index_r)>::value;

    //Need to restore the indices of the two tensors involved
    constexpr auto t1 = std::make_tuple(static_cast<size_t>(0));
    constexpr auto t2 = std::make_tuple(static_cast<size_t>(0));

    constexpr auto E1_size = std::tuple_size<typename E1::index_t>::value;
    constexpr auto E2_size = std::tuple_size<typename E2::index_t>::value;

    constexpr auto E1_p1 = get_subtuple<0,i1-1>(index_r);
    constexpr auto E1_p2 = get_subtuple<i1,E1_size-2>(index_r);

    constexpr auto E2_p1 = get_subtuple<E1_size-1,i2-1>(index_r);
    constexpr auto E2_p2 = get_subtuple<i2,E2_size-2>(index_r);

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
