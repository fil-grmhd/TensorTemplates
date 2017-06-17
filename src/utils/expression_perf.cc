#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <chrono>

#define TEMPLATES
#define ARRAYS
#define COMPARE

#include "../tensor_templates.hh"

int main(void) {
  using namespace tensors;
  constexpr size_t n = 30000000;
//  constexpr size_t n = 1000;


  // init random gens
  std::random_device rd;
//  std::minstd_rand gen(rd());
  std::mt19937_64 gen(rd());

  // uniform random number gen in 0,1
  std::uniform_real_distribution<> dist_uni(0,1);

  // "gridfunctions"
  std::array<double,n> axx;
  std::array<double,n> axy;
  std::array<double,n> axz;
  std::array<double,n> ayx;
  std::array<double,n> ayy;
  std::array<double,n> ayz;
  std::array<double,n> azx;
  std::array<double,n> azy;
  std::array<double,n> azz;

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

  std::cout << "Finished (~ " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count()
            << "s )." << std::endl;

#ifdef TEMPLATES

  using tensor_t = tensor3_t<double,lower_t,upper_t>;
  using resulting_tensor_t = tensor3_t<double,upper_t,lower_t>;
  using vector_t = vector3_t<double>;

  std::array<resulting_tensor_t,n> contracted_tensors;
  std::array<vector_t,n> contracted_vectors;

  std::array<double,n> traces;

  tensor_field_t<tensor_t> tensor_field(&axx[0],&ayx[0],&azx[0],
                                        &axy[0],&ayy[0],&azy[0],
                                        &axz[0],&ayz[0],&azz[0]);

  tensor_field_t<vector_t> vector_field(&axx[0],&ayy[0],&azz[0]);

  std::cout << "Computing templates..." << std::endl;

  // start timing
  t0 = std::chrono::high_resolution_clock::now();

  for(size_t i = 0; i<n; ++i) {
    // contracted tensor of dim = 3, rank = 2
    contracted_tensors[i] = contract<0,1>(tensor_field[i],tensor_field[i]);
    // contracted vector of dim = 3
    contracted_vectors[i] = contract<0,0>(tensor_field[i],vector_field[i]);

    // traces of rank = 2 tensors
    traces[i] = trace<0,1>(tensor_field[i]);
  }

  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished (~ " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count()
            << "s )." << std::endl;

#endif

#ifdef ARRAYS
  // contracted tensor of dim = 3, rank = 2
  std::array<double,n> bxx;
  std::array<double,n> bxy;
  std::array<double,n> bxz;
  std::array<double,n> byx;
  std::array<double,n> byy;
  std::array<double,n> byz;
  std::array<double,n> bzx;
  std::array<double,n> bzy;
  std::array<double,n> bzz;

  // contracted vector of dim = 3
  std::array<double,n> cx;
  std::array<double,n> cy;
  std::array<double,n> cz;

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

  std::cout << "Finished (~ " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count()
            << "s )." << std::endl;
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

  constexpr int exp = 14;

  for(size_t i = 0; i<n; ++i) {
    if(!resulting_tensor_t(array_field[i]).compare_components<exp>(contracted_tensors[i])) {
      std::cout << "Tensors differ at " << i << std::endl;
      std::cout << array_field[i] << std::endl;
      std::cout << contracted_tensors[i] << std::endl;
    }
    if(!vector_t(array_vector_field[i]).compare_components<exp>(contracted_vectors[i])) {
      std::cout << "Vectors differ at " << i << std::endl;
      std::cout << array_vector_field[i] << std::endl;
      std::cout << contracted_vectors[i] << std::endl;
    }
    if(d0[i] != traces[i]) {
      std::cout << "Trace differ at " << i << std::endl;
      std::cout << d0[i] << std::endl;
      std::cout << traces[i] << std::endl;
    }
  }

  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished (~ " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count()
            << "s )." << std::endl;

#endif
#endif
#endif
}
