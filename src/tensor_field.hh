#ifndef TENSOR_FIELD_HH
#define TENSOR_FIELD_HH

namespace tensors {

#ifdef TF_CREATOR

//! Template for generic tensor field
// A tensor field is not a tensor expression, but delivers tensors at different grid points.
// This tensors are then used in expression.
// One could make the tensor field also a tensor expression,
// but in that case one would need to store an internal state (the current grid point),
// which is really really bad for thread safety.
//template<typename E>
template<typename tensor_t>
//class tensor_field_t : public tensor_expression_t<tensor_field_t<E> > {
class tensor_field_t {
  public:
    // Get properties of underlying tensor type
    using property_t = typename tensor_t::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  private:
    //! Storage for ndof pointers
    std::array<data_t*,ndof> m_data{};

    //! Creates a tensor at (pointer) index i
    template<size_t... I>
    inline decltype(auto) get_tensor(size_t const i, std::index_sequence<I...>) const {
      return tensor_t((m_data[I][i])...);
    }
  public:
    //! Constructor from pointer parameters
    template <typename... TArgs>
    tensor_field_t(data_t * first_elem, TArgs... elem)
        : m_data({first_elem, elem...}) {
      static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
    };

    //! Returns a tensor at (pointer) index i
    inline decltype(auto) operator[](size_t const i) const {
      return get_tensor(i, std::make_index_sequence<ndof>{});
    }
};

#endif

#ifdef TF_EXPRESSION

//! Template for generic tensor field
template<typename E>
class tensor_field_t : public tensor_expression_t<tensor_field_t<E> > {
  public:
    // Get properties of underlying tensor type
    using property_t = typename E::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  private:
    //! Storage for ndof pointers
    std::array<data_t*,ndof> m_data{};

    //! Internal (pointer) index
    static size_t ptr_index;
    #pragma omp threadprivate(ptr_index)

  public:
    //! Constructor from pointer parameters
    template <typename... TArgs>
    tensor_field_t(data_t * first_elem, TArgs... elem)
        : m_data({first_elem, elem...}) {
      static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
    };

    static void move_to(size_t i) {
      ptr_index = i;
    }

    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline decltype(auto) operator[](size_t i) const = delete;

    template<size_t index>
    inline decltype(auto) evaluate() const {
      return m_data[index][ptr_index];
    }
};
// get memory for static member
template<typename E>
size_t tensor_field_t<E>::ptr_index;

#endif

}
#endif
