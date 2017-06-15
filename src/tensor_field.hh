#ifndef TENSOR_FIELD_HH
#define TENSOR_FIELD_HH

namespace tensors {

//! Template for generic tensor field expression
//! This represents a tensor at a specific grid position
template<typename T>
class tensor_field_expression_t : public tensor_expression_t<tensor_field_expression_t<T>> {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  private:
    //! Reference to pointer array of underlying tensor field
    std::array<data_t*,ndof> const & ptr_array;

    //! Internal (pointer) index
    size_t ptr_index;

  public:
    //! Contructor (called from a tensor field)
    tensor_field_expression_t(std::array<data_t*,ndof> const & arr, size_t const index)
        : ptr_array(arr), ptr_index(index) {}

    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline decltype(auto) operator[](size_t i) const = delete;

    template<size_t index>
    inline decltype(auto) evaluate() const {
      return ptr_array[index][ptr_index];
    }
};

//! Template for generic tensor field
//! A tensor field is not a tensor expression,
//! but delivers tensor field expressions at different grid points.
template<typename T>
class tensor_field_t {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  private:
    //! Storage for ndof pointers
    std::array<data_t*,ndof> ptr_array;

  public:
    //! Constructor from pointer parameters
    template <typename... TArgs>
    tensor_field_t(data_t * first_elem, TArgs... elem)
        : ptr_array({first_elem, elem...}) {
      static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
    };

    //! Returns a tensor field expression at (pointer) index i
    inline decltype(auto) operator[](size_t const i) const {
      return tensor_field_expression_t<T>(ptr_array,i);
    }

    //! Set the tensor field components at (pointer) index i
    inline void set_components(size_t const i, T const &t) {
      for(size_t k = 0; k<ndof; ++k)
        ptr_array[k][i] = t[k];
    }
};

}
#endif
