#include <iostream>

#define TENSORS_VECTORIZED
#define TENSORS_AUTOVEC
#include "../tensor_templates.hh"

using namespace tensors;

int main() {
  constexpr int deg = 1;
  constexpr int order = 2;

  using central_w_t = fd::fd_weights<deg,order,fd::central_nodes>;
  using side_pos_w_t = fd::fd_weights<deg,order,fd::onesided_nodes<false>>;
  using side_neg_w_t = fd::fd_weights<deg,order,fd::onesided_nodes<true>>;

  constexpr central_w_t central_w;
  constexpr side_pos_w_t side_pos_w;
  constexpr side_neg_w_t side_neg_w;

  std::cout << "Central nodes and weights:" << std::endl;
  for(int i = 0; i <= order; ++i) {
    std::cout << central_w_t::node_t::node(i) << " -> " << central_w.weight(i) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Onesided (pos characteristic) nodes and weights:" << std::endl;
  for(int i = 0; i <= order; ++i) {
    std::cout << side_pos_w_t::node_t::node(i) << " -> " << side_pos_w.weight(i) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Onesided (neg charateristic) nodes and weights:" << std::endl;
  for(int i = 0; i <= order; ++i) {
    std::cout << side_neg_w_t::node_t::node(i) << " -> " << side_neg_w.weight(i) << std::endl;
  }
  std::cout << std::endl;
}
