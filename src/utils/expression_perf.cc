#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <chrono>
#include <memory>

#define TEMPLATES
#define ARRAYS
#define COMPARE

#include "../tensor_templates.hh"

int main(void) {
  using namespace tensors;
//  constexpr size_t n = 30000000;
//  constexpr size_t n = 100000;
  constexpr size_t n = 1000;


  // init random gens
  std::random_device rd;
//  std::minstd_rand gen(rd());
  std::mt19937_64 gen(rd());

  // uniform random number gen in 0,1
  std::uniform_real_distribution<> dist_uni(0,1);

  // "gridfunctions"
  auto axx = std::make_unique<double[]>(n);
  auto axy = std::make_unique<double[]>(n);
  auto axz = std::make_unique<double[]>(n);
  auto ayx = std::make_unique<double[]>(n);
  auto ayy = std::make_unique<double[]>(n);
  auto ayz = std::make_unique<double[]>(n);
  auto azx = std::make_unique<double[]>(n);
  auto azy = std::make_unique<double[]>(n);
  auto azz = std::make_unique<double[]>(n);

  std::cout << "Filling GFs..." << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();
  // fill 'em
  for(size_t i = 0; i<n; ++i) {

    axx[i] = dist_uni(gen);
    axy[i] = dist_uni(gen);
    axz[i] = dist_uni(gen);

    ayx[i] = dist_uni(gen);
    ayy[i] = dist_uni(gen);
    ayz[i] = dist_uni(gen);

    azx[i] = dist_uni(gen);
    azy[i] = dist_uni(gen);
    azz[i] = dist_uni(gen);

/*
    axx[i] = 1;
    axy[i] = 2;
    axz[i] = 3;

    ayx[i] = 4;
    ayy[i] = 5;
    ayz[i] = 6;

    azx[i] = 7;
    azy[i] = 8;
    azz[i] = 9;
*/
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished (~ " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
            << " ms )." << std::endl;

#ifdef TEMPLATES

  using tensor_t = tensor3_t<double,lower_t,upper_t>;
  using resulting_tensor_t = tensor3_t<double,upper_t,lower_t>;
  using vector_t = vector3_t<double>;

  // test result "gridfunctions"
  auto trxx = std::make_unique<double[]>(n);
  auto trxy = std::make_unique<double[]>(n);
  auto trxz = std::make_unique<double[]>(n);
  auto tryx = std::make_unique<double[]>(n);
  auto tryy = std::make_unique<double[]>(n);
  auto tryz = std::make_unique<double[]>(n);
  auto trzx = std::make_unique<double[]>(n);
  auto trzy = std::make_unique<double[]>(n);
  auto trzz = std::make_unique<double[]>(n);


  auto vrx = std::make_unique<double[]>(n);
  auto vry = std::make_unique<double[]>(n);
  auto vrz = std::make_unique<double[]>(n);

  // output tensor fields
  tensor_field_t<resulting_tensor_t> contracted_tensors(&trxx[0],&tryx[0],&trzx[0],
                                                        &trxy[0],&tryy[0],&trzy[0],
                                                        &trxz[0],&tryz[0],&trzz[0]);

  tensor_field_t<vector_t> contracted_vectors(&vrx[0],&vry[0],&vrz[0]);

// these are always (slightly) faster on assignement, probably due to move semantics
//  std::array<resulting_tensor_t,n> contracted_tensors;
//  std::array<vector_t,n> contracted_vectors;

  std::array<double,n> traces;

  // random input tensor fields
  tensor_field_t<tensor_t> tensor_field(&axx[0],&ayx[0],&azx[0],
                                        &axy[0],&ayy[0],&azy[0],
                                        &axz[0],&ayz[0],&azz[0]);

  tensor_field_t<vector_t> vector_field(&axx[0],&ayy[0],&azz[0]);

  std::cout << "Computing templates..." << std::endl;

  // start timing
  t0 = std::chrono::high_resolution_clock::now();

  for(size_t i = 0; i<n; ++i) {
    // contracted tensor of dim = 3, rank = 2
    // evaluation is triggered in set_components
    auto contracted_tensor = contract<0,1>(tensor_field[i],tensor_field[i]);

// more complicated expression
//    auto contracted_tensor = contract<0,0>(tensor_field[i],
//                                           contract<0,1>(tensor_field[i],
//                                                         tensor_field[i]));

  //  contracted_tensors.set_components(i,contracted_tensor);
    contracted_tensors[i] = contracted_tensor;

// slightly faster, prob move semantics
//    contracted_tensors[i] = contract<0,1>(tensor_field[i],tensor_field[i]);

    // contracted vector of dim = 3
    // evaluation is triggered in set_components
    auto contracted_vector = contract<0,0>(tensor_field[i],vector_field[i]);
    //contracted_vectors.set_components(i,contracted_vector);
    contracted_vectors[i] = contracted_vector;
// slightly faster, prob move semantics
//    contracted_vectors[i] = contract<0,0>(tensor_field[i],vector_field[i]);

    // traces of rank = 2 tensors
    traces[i] = trace<0,1>(tensor_field[i]);
  }

  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished (~ " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
            << " ms )." << std::endl;

#endif

#ifdef ARRAYS
  // contracted tensor of dim = 3, rank = 2
  auto bxx = std::make_unique<double[]>(n);
  auto bxy = std::make_unique<double[]>(n);
  auto bxz = std::make_unique<double[]>(n);
  auto byx = std::make_unique<double[]>(n);
  auto byy = std::make_unique<double[]>(n);
  auto byz = std::make_unique<double[]>(n);
  auto bzx = std::make_unique<double[]>(n);
  auto bzy = std::make_unique<double[]>(n);
  auto bzz = std::make_unique<double[]>(n);

  // contracted vector of dim = 3

  auto cx = std::make_unique<double[]>(n);
  auto cy = std::make_unique<double[]>(n);
  auto cz = std::make_unique<double[]>(n);

  // traces of tensors
  std::array<double,n> d0;

  std::cout << "Contracting arrays..." << std::endl;

  // start timing
  t0 = std::chrono::high_resolution_clock::now();

  for(size_t i = 0; i<n; ++i) {
    // contracted tensor components
    bxx[i] = axx[i]*axx[i]
            +ayx[i]*axy[i]
            +azx[i]*axz[i];
    bxy[i] = axx[i]*ayx[i]
            +ayx[i]*ayy[i]
            +azx[i]*ayz[i];
    bxz[i] = axx[i]*azx[i]
            +ayx[i]*azy[i]
            +azx[i]*azz[i];

    byx[i] = axy[i]*axx[i]
            +ayy[i]*axy[i]
            +azy[i]*axz[i];
    byy[i] = axy[i]*ayx[i]
            +ayy[i]*ayy[i]
            +azy[i]*ayz[i];
    byz[i] = axy[i]*azx[i]
            +ayy[i]*azy[i]
            +azy[i]*azz[i];

    bzx[i] = axz[i]*axx[i]
            +ayz[i]*axy[i]
            +azz[i]*axz[i];
    bzy[i] = axz[i]*ayx[i]
            +ayz[i]*ayy[i]
            +azz[i]*ayz[i];
    bzz[i] = axz[i]*azx[i]
            +ayz[i]*azy[i]
            +azz[i]*azz[i];

    // contracted vector components
    cx[i] = axx[i]*axx[i]
           +ayx[i]*ayy[i]
           +azx[i]*azz[i];

    cy[i] = axy[i]*axx[i]
           +ayy[i]*ayy[i]
           +azy[i]*azz[i];

    cz[i] = axz[i]*axx[i]
           +ayz[i]*ayy[i]
           +azz[i]*azz[i];

    // traces of tensors
    d0[i] = axx[i] + ayy[i] + azz[i];
  }

  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished (~ " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
            << " ms )." << std::endl;
#endif

#ifdef COMPARE
#ifdef TEMPLATES
#ifdef ARRAYS
  std::cout << "Testing equality..." << std::endl;

  t0 = std::chrono::high_resolution_clock::now();


  tensor_field_t<resulting_tensor_t> array_field(&bxx[0],&byx[0],&bzx[0],
                                                 &bxy[0],&byy[0],&bzy[0],
                                                 &bxz[0],&byz[0],&bzz[0]);

  tensor_field_t<vector_t> array_vector_field(&cx[0],&cy[0],&cz[0]);

  constexpr int exp = 15;

  for(size_t i = 0; i<n; ++i) {
    if(!resulting_tensor_t(array_field[i]).compare_components<exp>(contracted_tensors[i]).first) {
      std::cout << "Tensors differ at " << i << std::endl;
      std::cout << array_field[i] << std::endl;
      std::cout << contracted_tensors[i] << std::endl;
      std::cout << resulting_tensor_t(array_field[i]).compare_components<exp>(contracted_tensors[i]).second << std::endl;
    }
    if(!vector_t(array_vector_field[i]).compare_components<exp>(contracted_vectors[i]).first) {
      std::cout << "Vectors differ at " << i << std::endl;
      std::cout << array_vector_field[i] << std::endl;
      std::cout << contracted_vectors[i] << std::endl;
      std::cout << vector_t(array_vector_field[i]).compare_components<exp>(contracted_vectors[i]).second << std::endl;
    }
    if(d0[i] != traces[i]) {
      std::cout << "Trace differ at " << i << std::endl;
      std::cout << d0[i] << std::endl;
      std::cout << traces[i] << std::endl;
    }
  }

  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished (~ " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
            << " ms )." << std::endl;

#endif
#endif
#endif
}
