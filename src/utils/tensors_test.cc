#include<iostream>

//#define TENSORS_VECTORIZED
#include "../tensor_templates.hh"

struct op_drop {
  template<typename... Args>
  inline static int op(Args&&... args) {
    return 0;
  }
};

template<typename F, int min, int max>
struct index_recursion_t {
  template<int t_ind, typename OP>
  struct traverser_t {
    template<typename... Args>
    inline static decltype(auto) traverse(Args&&... args) {
      return OP::op(F::template call<t_ind>(std::forward<Args>(args)...),
                    traverser_t<t_ind+1,OP>::traverse(std::forward<Args>(args)...));
    }
  };
  template<typename OP>
  struct traverser_t<max,OP> {
    template<typename... Args>
    inline static decltype(auto) traverse(Args&&... args) {
      return F::template call<max>(std::forward<Args>(args)...);
    }
  };

  template<typename OP = op_drop, typename... Args>
  inline static decltype(auto) traverse(Args&&... args) {
    return traverser_t<min,OP>::traverse(std::forward<Args>(args)...);
  }
};

template<int cind, typename tensor_t>
struct uncompressed_index_printer : public index_recursion_t<uncompressed_index_printer<cind,tensor_t>,0,tensor_t::rank-1> {
  template<int iind>
  inline static decltype(auto) call() {
    std::cout << tensor_t::property_t::symmetry_t::template uncompress_index<iind,cind>::value;
    if(iind != tensor_t::rank-1)
      std::cout << ",";
    return 0;
  }
};


template<typename T, typename U>
struct index_compare_printer_t : public index_recursion_t<index_compare_printer_t<T,U>,0,T::ndof-1> {
  template<int cind>
  inline static decltype(auto) call() {
    std::cout << "[" << cind << "]: ";
    std::cout << "(";
    constexpr int sym_cind0 = T::symmetry_t::template index_from_generic<cind>::value;
    uncompressed_index_printer<sym_cind0,T>::traverse();
    std::cout << ") <-> (";
    constexpr int sym_cind1 = U::symmetry_t::template index_from_generic<cind>::value;
    uncompressed_index_printer<sym_cind1,U>::traverse();
    std::cout << ")" << std::endl;
    return 0;
  }
};

int main(){

  using namespace tensors;

  using sym3_01_t = sym_tensor3_t<double,0,1,lower_t,lower_t,lower_t>;
  using sym4_01_t = sym_tensor3_t<double,0,1,lower_t,lower_t,lower_t,lower_t>;
  using sym3_12_t = sym_tensor3_t<double,1,2,lower_t,lower_t,lower_t>;
  using sym4_12_t = sym_tensor3_t<double,1,2,lower_t,lower_t,lower_t,lower_t>;
  using sym4_23_t = sym_tensor3_t<double,2,3,lower_t,lower_t,lower_t,lower_t>;
  using sym4_02_t = sym_tensor3_t<double,0,2,lower_t,lower_t,lower_t,lower_t>;
  using sym4_03_t = sym_tensor3_t<double,0,3,lower_t,lower_t,lower_t,lower_t>;
  using sym4_13_t = sym_tensor3_t<double,1,3,lower_t,lower_t,lower_t,lower_t>;

  using t_type0 = tensor3_t<double,lower_t,lower_t,lower_t,lower_t>;
  using t_type1 = sym4_03_t;

  index_compare_printer_t<t_type0,t_type1>::traverse();
}



