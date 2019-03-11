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

template<typename T, size_t compressed_index, size_t current_index>
//struct index_signs : public index_recursion_constexpr_t<index_signs<T,compressed_index,current_index>,current_index,T::property_t::rank-1> {
struct index_signs : public index_recursion_t<index_signs<T,compressed_index,current_index>,current_index+1,T::property_t::rank-1> {
  template<int ind>
  inline static decltype(auto) call() {
    std::cout << "Comparing " << current_index << " to " << ind << std::endl;

    constexpr int sym_ind = T::property_t::symmetry_t::template index_from_generic<compressed_index>::value;
    constexpr int current_val = T::property_t::symmetry_t::template uncompress_index<current_index,sym_ind>::value;
    constexpr int index_val = T::property_t::symmetry_t::template uncompress_index<ind,sym_ind>::value;

    constexpr int diff = index_val - current_val;
    std::cout << "Difference:" << index_val << " - " << current_val << " = " << diff << std::endl;
    std::cout << "Signum:" << utilities::static_sign<diff>::value << std::endl;

    return utilities::static_sign<diff>::value;
  }
};

template<typename T, size_t cind>
struct index_loop : public index_recursion_t<index_loop<T,cind>,0,T::rank-2> {

  template<int ind>
  inline static decltype(auto) call() {
    std::cout << "Starting inner loop on compress index " << cind << " and index " << ind << std::endl;
    return index_signs<T,cind,ind>::template traverse<op_mult>();
  }
};

int main(){
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

  constexpr size_t inds[] = {1,0,2,1};
  constexpr size_t cind_gen = gen_type::symmetry_t::compressed_index<inds[0], inds[1], inds[2], inds[3]>::value;
  constexpr size_t cind_sym = sym_type::symmetry_t::compressed_index<inds[0], inds[1], inds[2], inds[3]>::value;

  std::cout << "Calling with compressed indices " << cind_gen << " " << cind_sym << std::endl;
  std::cout << "And uncompressed indices " << inds[0] << " " << inds[1] << " " << inds[2] << " " << inds[3] << std::endl;

  std::cout << "Signum multiplication gen: " << std::endl << index_loop<gen_type,cind_gen>::traverse<op_mult>() << std::endl;
  std::cout << "Signum multiplication sym: " << std::endl << index_loop<sym_type,cind_sym>::traverse<op_mult>() << std::endl;
}



