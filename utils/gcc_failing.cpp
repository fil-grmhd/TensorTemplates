#include <iostream>
#include "../tensor_templates.hh"

using namespace tensors;

int main() {
  sym_tensor3_t<double,0,1,upper_t,upper_t> w;
  tensor3_t<double,lower_t,lower_t> x;

  auto wx_contracted = contract(w,x);
  auto sym_wx_contracted = sym2_cast(wx_contracted);
  auto sym_wx_contracted_both = sym2_cast(contract(w,x));

  // fine
  std::cout << evaluate(wx_contracted) << std::endl;
  std::cout << evaluate(sym_wx_contracted) << std::endl;
  // fail
  std::cout << evaluate(sym_wx_contracted_both) << std::endl;

  vector4_t<double> u(1,2,3,4);
  vector4_t<double> v(1,2,3,4);

  auto uu = tensor_cat(u,u);
  auto sym_uu = sym2_cast(uu);
  auto sym_uu_test0 = sym2_cast(tensor_cat(u,u));
  auto sym_uu_test1 = sym2_cast(tensor_cat(u,v));

  // fine
  std::cout << evaluate(uu) << std::endl;
  std::cout << evaluate(sym_uu) << std::endl;
  // reading from obscure memory locations (on my laptop)
  std::cout << evaluate(sym_uu_test0) << std::endl;
  std::cout << evaluate(sym_uu_test1) << std::endl;

}
