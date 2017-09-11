#include<iostream>

//#define TENSORS_VECTORIZED
#include "../tensor_templates.hh"

double rel_diff(double d0, double d1) {
  return 2 * std::abs(d0 - d1)
         / (std::abs(d0) + std::abs(d1));
}
constexpr double eps = 1e-14;


template<typename node_t, size_t order>
struct test_diff {
  const double idx;
  const int stride[3];

  test_diff() : idx(1), stride {1,1,1} {}

  template<size_t dir, typename T>
  inline __attribute__ ((always_inline)) T diff(T const * const ptr, size_t const index) const {
    return idx*tensors::fd::auto_diff<1,order,node_t>(ptr, index, stride[dir]);
  }
};


struct op_drop {
  template<typename... Args>
  inline static decltype(auto) op(Args&&... args) {
    return 0;
  }
};

struct op_and {
  template<typename Arg, typename... Args>
  inline static decltype(auto) op(Arg arg, Args... args) {
    return arg && op(args...);
  }
  template<typename Arg>
  inline static decltype(auto) op(Arg arg) {
    return arg;
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
    constexpr int sym_cind = tensor_t::symmetry_t::template index_from_generic<cind>::value;

    if(iind == 0)
      std::cout << "(";
    std::cout << tensor_t::property_t::symmetry_t::template uncompress_index<iind,sym_cind>::value;
    if(iind != tensor_t::rank-1)
      std::cout << ",";
    else
      std::cout << ")";
    return 0;
  }
};

template<int cind, typename T, typename S, int i0, int i1>
struct uncompressed_index_comparison : public index_recursion_t<uncompressed_index_comparison<cind,T,S,i0,i1>,0,T::rank-1> {
  template<int iind>
  inline static decltype(auto) call() {
    constexpr int sym_cind0 = T::symmetry_t::template index_from_generic<cind>::value;
    constexpr int sym_cind1 = S::symmetry_t::template index_from_generic<cind>::value;

    constexpr int ind0 = T::property_t::symmetry_t::template uncompress_index<iind,sym_cind0>::value;
    constexpr int ind1 = S::property_t::symmetry_t::template uncompress_index<iind,sym_cind1>::value;

    if(iind == i0 || iind == i1) {
      constexpr int s_ind0 = S::property_t::symmetry_t::template uncompress_index<i0,sym_cind1>::value;
      constexpr int s_ind1 = S::property_t::symmetry_t::template uncompress_index<i1,sym_cind1>::value;
      return (ind0 == s_ind0 || ind0 == s_ind1);
    }
    else
      return (ind0 == ind1);
  }
};

template<typename T, typename S, int i0, int i1, bool print = false>
struct sym_to_gen_comp : public index_recursion_t<sym_to_gen_comp<T,S,i0,i1,print>,0,T::ncomp-1> {
  template<int cind>
  inline static decltype(auto) call(T const & t, S const & s) {
    if(!uncompressed_index_comparison<cind,T,S,i0,i1>::template traverse<op_and>()) {
      if(print) {
        std::cout << "index diff [" << cind << "]: ";
        uncompressed_index_printer<cind,T>::traverse();
        std::cout << " != ";
        uncompressed_index_printer<cind,S>::traverse();
        std::cout << std::endl;
      }
      return false;
    }
    if(t.template evaluate<cind>() != s.template evaluate<cind>()) {
      if(print) {
        std::cout << "comp diff [" << cind << "],";
        uncompressed_index_printer<cind,T>::traverse();
        std::cout << ",";
        uncompressed_index_printer<cind,S>::traverse();
        std::cout << ": ";
        std::cout << t.template evaluate<cind>() << " != " << s.template evaluate<cind>() << std::endl;
      }
      return false;
    }
    return true;
  }
};

template<typename T, typename S, bool print = false>
struct gen_to_gen_comp : public index_recursion_t<gen_to_gen_comp<T,S,print>,0,T::ncomp-1> {
  template<int cind>
  inline static decltype(auto) call(T const & t, S const & s) {
    if(t.template evaluate<cind>() != s.template evaluate<cind>()) {
      if(print) {
        std::cout << "comp diff [" << cind << "],";
        uncompressed_index_printer<cind,T>::traverse();
        std::cout << ",";
        uncompressed_index_printer<cind,S>::traverse();
        std::cout << ": ";
        std::cout << t.template evaluate<cind>() << " != " << s.template evaluate<cind>() << std::endl;
      }
      return false;
    }
    return true;
  }
};

template<typename T, bool print = false>
struct check_slopes : public index_recursion_t<check_slopes<T,print>,0,T::ncomp-1> {
  template<int cind, typename V>
  inline static decltype(auto) call(T const & t, V const & slopes) {
    constexpr int sym_cind = T::symmetry_t::template index_from_generic<cind>::value;
    constexpr int comp_index = T::symmetry_t::template uncompress_index<0,sym_cind>::value;

    double rel_err = rel_diff(t.template evaluate<cind>(),slopes.template evaluate<comp_index>());
    if(rel_err > eps) {
      if(print) {
        std::cout << "slope wrong [" << cind << "],";
        uncompressed_index_printer<cind,T>::traverse();
        std::cout << "," << comp_index;
        std::cout << ": ";
        std::cout << t.template evaluate<cind>() << " != " << slopes.template evaluate<comp_index>() << " (" << rel_err << ")" << std::endl;
      }
      return false;
    }
    return true;
  }
};


int main(){

  using namespace tensors;

  constexpr int i0 = 0;
  constexpr int i1 = 2;

  using sym_type = sym_tensor3_t<double,i0,i1,lower_t,lower_t,lower_t,lower_t>;
  using gen_type = tensor3_t<double,lower_t,lower_t,lower_t,lower_t>;

  sym_type sym;
  for(int i = 0; i<sym_type::ndof; i++)
    sym[i] = i+1;

  gen_type gen;

  gen = sym;
  //sym = gen;

  auto cast = evaluate(sym2_cast<i0,i1>(gen));

  std::cout << "Comparing sym to gen... " << sym_to_gen_comp<gen_type,sym_type,i0,i1>::traverse<op_and>(gen,sym) << std::endl;
  std::cout << "Comparing sym to cast... " << sym_to_gen_comp<decltype(cast),sym_type,i0,i1>::traverse<op_and>(cast,sym) << std::endl;

  for(int i = 0; i < gen_type::ndof; ++i)
    gen[i] = i+1;

  auto gen_reorder0 = reorder_index<1,0,2,3>(gen);
  auto gen_reorder1 = reorder_index<0,1,3,2>(gen_reorder0);
  auto gen_reorder2 = reorder_index<0,2,1,3>(gen_reorder1);
  auto gen_reorder3 = evaluate(reorder_index<2,0,3,1>(gen_reorder2));

  std::cout << "Comparing gen to reordered... " << gen_to_gen_comp<gen_type,gen_type>::traverse<op_and>(gen,gen_reorder3) << std::endl;

  auto sym_reorder0 = reorder_index<1,0,2,3>(sym);
  auto sym_reorder1 = reorder_index<0,1,3,2>(sym_reorder0);
  auto sym_reorder2 = reorder_index<0,2,1,3>(sym_reorder1);
//  auto sym_reorder3 = evaluate(sym2_cast<i0,i1>(reorder_index<2,0,3,1>(sym_reorder2)));
  auto sym_reorder3 = evaluate(reorder_index<2,0,3,1>(sym_reorder2));

  std::cout << "Comparing sym to reordered... " << gen_to_gen_comp<sym_type,gen_type>::traverse<op_and>(sym,sym_reorder3) << std::endl;

  constexpr int num_points = 20;
  double vec0[num_points], vec1[num_points], vec2[num_points];

  vector3_t<double> slopes(2,10,1337);
  vector3_t<double> constants(5,16,42);

  for(int i = 0; i<num_points; ++i) {
    vec0[i] = slopes[0]*i+constants[0];
    vec1[i] = slopes[1]*i+constants[1];
    vec2[i] = slopes[2]*i+constants[2];
  }

  test_diff<fd::central_nodes,4> c4_diff;
  test_diff<fd::onesided_nodes<false,1>,4> up4_diff;
  test_diff<fd::onesided_nodes<true,1>,4> down4_diff;

  tensor_field_t<vector3_t<double>> vec_field(vec0,vec1,vec2);
  auto dvec0 = evaluate(vec_field[num_points/2].finite_diff(c4_diff));
  auto dvec1 = evaluate(vec_field[num_points/2].finite_diff(up4_diff));
  auto dvec2 = evaluate(vec_field[num_points/2].finite_diff(down4_diff));

  std::cout << "Comparing slopes (central)... " << check_slopes<decltype(dvec0)>::traverse<op_and>(dvec0,slopes) << std::endl;
  std::cout << "Comparing slopes (up)... " << check_slopes<decltype(dvec1)>::traverse<op_and>(dvec1,slopes) << std::endl;
  std::cout << "Comparing slopes (down)... " << check_slopes<decltype(dvec2)>::traverse<op_and>(dvec2,slopes) << std::endl;
}



