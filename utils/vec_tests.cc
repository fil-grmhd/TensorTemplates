#include <vector>
#include <iostream>

#define TENSORS_VECTORIZED
#include "../tensor_templates.hh"

using namespace tensors;

int main() {
  using sca_type = double;
  using vec_type = Vc::Vector<sca_type>;

  constexpr size_t N = 10000;
  constexpr size_t vec_size = vec_type::Size;
  constexpr size_t num_data = N*vec_size;
  constexpr size_t order = 4;
  constexpr size_t stride = 2;

  // some GF
  std::vector<sca_type> data_0(num_data);

  std::vector<sca_type> deriv(num_data);

  for(int k = 0; k < N; ++k) {
    vec_type derivative;

    sca_type const * const grid_ptr = &data_0[k*vec_type::Size];
    sca_type * const deriv_ptr = &deriv[k*vec_type::Size];

    for(int i = 0; i < order + 1; ++i) {
      for(int j = 0; j < vec_type::Size; ++j) {
        sca_type const * const vec_grid_pointer = grid_ptr + j;

        derivative[j] += vec_grid_pointer[(order - order / 2 - i) * stride] * fd::fd_stencils<order>::stencil[order / 2][order];
      }
    }
    // "1.0/dx"
    derivative *= 1.0/0.1;
    derivative.store(deriv_ptr);
  }

  for(int i = 0; i < num_data; i += num_data/10) {
    std::cout << deriv[i] << std::endl;
  }
}
