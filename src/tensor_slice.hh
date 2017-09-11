#ifndef TENSOR_SLICE_HH
#define TENSOR_SLICE_HH

namespace tensors {

//! Expression template for generic tensor expression slice
template <typename E, int... Ind>
class tensor_slice_t : public tensor_expression_t<tensor_slice_t<E, Ind...>> {
protected:
  // references to sliced expression
  E const &_u;

  // can only shift dimensions up to a two component (co)vector
  // a single component can be accessed directly...
  static constexpr size_t max_shift = E::property_t::ndim-2;

public:
  static_assert(utilities::all_true<
                  ((Ind >= -static_cast<int>(max_shift + 1)) &&
                   (Ind < static_cast<int>(E::property_t::ndim)))...
                >::value,
                "Indices out of range. Use 0..ndim-1 to fix indices "
                "and -1...-(ndim - 2)-1 to indicate a (shifted) free index.");

  static constexpr size_t rank = count_free_indices<Ind...>::value;
  static constexpr size_t ndim = E::property_t::ndim - free_index_shift<Ind...>::value;

  // The following is slightly cumbersome:
  // We encode the slice by <0 -1 2> which would be equivalent
  // to A_{0 \mu 2} (note that -1 refers to a free index)
  // The compressed index for this is 1d i.e. i_j but
  // the compressed index of the full tensor requires us to insert
  // 0 and 2 and to add i_j with the correct power of ndim according to
  // it's position.
  // This happens here: N is Nth -1 index in the parameter pack.
  // to get the position of this Nth index in the parameter pack, e.g. 1
  // in the above example, we count count upwards until it has been found.
  // The number of free indices is encoded in N_count. If we exceed N,
  // we must not increment the index further in the recursion, hence the
  // conditionals.

  template <size_t N, size_t N_count, int i0, int... Indices>
  struct get_free_indices {
    static constexpr size_t value =
        (N + 1 > N_count) +
        get_free_indices<N, N_count + (i0 == -1), Indices...>::value;
  };

  template <size_t N, size_t N_count, int i0>
  struct get_free_indices<N, N_count, i0> {
    static constexpr size_t value = (N + 1 > N_count) - 1;
  };

  template <size_t N> struct get_index_t {
    using type =
        typename std::tuple_element<get_free_indices<N, 0, Ind...>::value,
                                    typename E::property_t::index_t>::type;
  };

  template <size_t... Ind_slice> struct slice_index_t {
    using type = std::tuple<typename get_index_t<Ind_slice>::type...>;
  };

  template <size_t... Ind_slice>
  static constexpr decltype(auto)
  slice_index_t_wrapper(std::index_sequence<Ind_slice...>) {
    return typename slice_index_t<Ind_slice...>::type{};
  };

  template <typename Ind_s_t =
                std::make_index_sequence<count_free_indices<Ind...>::value>>
  static constexpr decltype(auto) slice_index_t_wrapper_wrapper() {
    return slice_index_t_wrapper(Ind_s_t{});
  };

  using property_t = general_tensor_property_t<general_tensor_t<
      typename E::property_t::data_t, typename E::property_t::frame_t,
      generic_symmetry_t<ndim,rank>, rank,
      decltype(slice_index_t_wrapper_wrapper()), ndim>>;

  tensor_slice_t(E const &u) : _u(u){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t c_index>
  inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {

    return _u.template evaluate<
                         compute_unsliced_cindex<
                           E,
                           typename property_t::this_tensor_t,
                           c_index,
                           0,
                           Ind...
                         >::value
                       >();
  }
};

template <int... Indices, typename E>
tensor_slice_t<E, Indices...> const inline __attribute__ ((always_inline)) slice(E const &u) {
  return tensor_slice_t<E, Indices...>(u);
};




template <typename T, typename frame_t_, typename symmetry_t_, size_t rank_, typename index_t_,
          size_t ndim_, int... Ind>
class tensor_assign_slice_t : public tensor_slice_t<general_tensor_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_>, Ind...>{

  //Inherit constructor
  using tensor_slice_t<general_tensor_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_>, Ind...>::tensor_slice_t;
  using super = tensor_slice_t<general_tensor_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_>, Ind...>;

  using this_tensor_t = tensor_assign_slice_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_,Ind...>;

public:
  //! Get component reference at (generic) compressed index position
  //  Needed if one wants to assign something to a specific element,
  //  given a generic compressed index.
  template<size_t index>
  inline __attribute__ ((always_inline)) T & cc() {
    return const_cast<general_tensor_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_> &>(super::_u).
      			template cc<
                         compute_unsliced_cindex<
                           general_tensor_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_>,
                           typename super::property_t::this_tensor_t,
                           index,
                           0,
                           Ind...
                         >::value
                       >();
  }


  //Add expression to tensor
  template <typename E>
  inline __attribute__ ((always_inline)) void operator+=(tensor_expression_t<E> const &tensor_expression){
//    static_assert(std::is_same<frame_t, typename E::property_t::frame_t>::value,
//                  "Frame types don't match!");

    static_assert(super::property_t::ndim== E::property_t::ndim, "Dimensions don't match!");

    static_assert(super::property_t::rank == E::property_t::rank, "Ranks don't match!");

    static_assert(std::is_same<T, typename E::property_t::data_t>::value,
                  "Data types don't match!");

    static_assert(
        compare_index_types<typename super::property_t::index_t, typename E::property_t::index_t,
                                 super::property_t::rank>(),
        "Index types do not match (e.g. lower_t != upper_t)!");

   add_to_tensor_t<E,this_tensor_t,super::property_t::ndof-1>::add_to_tensor(tensor_expression,*this);

  };

  //Add expression to tensor
  template <typename E>
  inline __attribute__ ((always_inline)) void operator-=(tensor_expression_t<E> const &tensor_expression){
//    static_assert(std::is_same<frame_t, typename E::property_t::frame_t>::value,
//                  "Frame types don't match!");

    static_assert(super::property_t::ndim == E::property_t::ndim, "Dimensions don't match!");

    static_assert(super::property_t::rank == E::property_t::rank, "Ranks don't match!");

    static_assert(std::is_same<T, typename E::property_t::data_t>::value,
                  "Data types don't match!");

    static_assert(
        compare_index_types<typename super::property_t::index_t, typename E::property_t::index_t,
                                 super::property_t::rank>(),
        "Index types do not match (e.g. lower_t != upper_t)!");

   subtract_from_tensor_t<E,this_tensor_t,super::property_t::ndof-1>::subtract_from_tensor(tensor_expression,*this);

  };


  //Multiply tensor with scalar
  inline __attribute__ ((always_inline)) void operator*=(T const & lambda){

   multiply_tensor_with_t<this_tensor_t,super::property_t::ndof-1>::multiply_tensor_with(lambda,*this);

  };

};

template <int... Ind, typename T, typename frame_t_, typename symmetry_t_, size_t rank_, typename index_t_,size_t ndim_>
inline __attribute__ ((always_inline)) decltype(auto) assign_slice(general_tensor_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_> & t){
  return tensor_assign_slice_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_, Ind...>(t);
};


}
#endif
