#include<iostream>

//#define TENSORS_VECTORIZED
#include "../src/tensor_templates.hh"

// lazy
using namespace tensors;

double rel_diff(double d0, double d1) {
  return 2 * std::abs(d0 - d1)
         / (std::abs(d0) + std::abs(d1));
}
constexpr double eps = 1e-20;
constexpr int stride_const = 3;

template<size_t order, size_t stencil_size, typename node_t>
struct test_diff {
  const double idx;
  const int stride[3];

  test_diff() : idx(1), stride {1,stride_const,stride_const*stride_const} {}

  template<size_t dir, typename T>
  inline __attribute__ ((always_inline)) T diff(T const * const ptr, size_t const index) const {
    return idx*tensors::fd::auto_diff<order,stencil_size,node_t>(ptr, index, stride[dir]);
  }
};


template<int cind, typename tensor_t>
struct uncompressed_index_printer : public index_recursion_t<uncompressed_index_printer<cind,tensor_t>,0,tensor_t::rank-1> {
  template<int iind>
  inline static decltype(auto) call() {
    constexpr int sym_cind = tensor_t::symmetry_t::template index_from_generic<cind>::value;

    std::cout << " !" << iind << "! ";
//    if(iind == 0)
//      std::cout << "(";
    std::cout << tensor_t::property_t::symmetry_t::template uncompress_index<iind,sym_cind>::value;
//    if(iind != tensor_t::rank-1)
//      std::cout << ",";
//    else
//      std::cout << ")";
    return 0;
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
  constexpr int num_points = 20*stride_const;
  double vec0[num_points], vec1[num_points], vec2[num_points];

  vector3_t<double> slopes(2,10,1337);
  vector3_t<double> constants(5,16,42);

  for(int i = 0; i<num_points; ++i) {
    vec0[i] = slopes[0]*i+constants[0];
    vec1[i] = slopes[1]*i+constants[1];
    vec2[i] = slopes[2]*i+constants[2];
  }

  test_diff<1,4,fd::central_nodes> c4_diff;
  test_diff<1,4,fd::onesided_nodes<false,1>> up4_diff;
  test_diff<1,4,fd::onesided_nodes<true,1>> down4_diff;

  tensor_field_t<vector3_t<double>> vec_field(vec0,vec1,vec2);
  auto dvec0 = evaluate(vec_field[num_points/2].finite_diff(c4_diff));
  auto dvec1 = evaluate(vec_field[num_points/2].finite_diff(up4_diff));
  auto dvec2 = evaluate(vec_field[num_points/2].finite_diff(down4_diff));

  std::cout << "Comparing slopes (central)... " << check_slopes<decltype(dvec0),true>::traverse<op_and>(dvec0,slopes) << std::endl;
  std::cout << "Comparing slopes (up)... " << check_slopes<decltype(dvec1),true>::traverse<op_and>(dvec1,slopes) << std::endl;
  std::cout << "Comparing slopes (down)... " << check_slopes<decltype(dvec2),true>::traverse<op_and>(dvec2,slopes) << std::endl;
}



