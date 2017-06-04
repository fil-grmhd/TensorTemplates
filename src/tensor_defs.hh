/*
 * =====================================================================================
 *
 *       Filename:  tensor_defs.hh
 *
 *    Description:  Some definitions 
 *
 *        Version:  1.0
 *        Created:  04/06/2017 19:18:05
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), emost@itp.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#ifndef TENSOR_DEFS_HH
#define TENSOR_DEFS_HH

namespace tensors {

struct comoving_t;
struct eulerian_t;

class rank_t {};
class upper_t : public rank_t {};
class lower_t : public rank_t {};

}

#endif

