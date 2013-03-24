#ifndef _UNYQUE_UBLAS_
#define _UNYQUE_UBLAS_

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace ublas = boost::numeric::ublas;
using namespace std;

namespace unyque {

  typedef ublas::vector<int> IVector;
  typedef ublas::zero_vector<int> IVector_zero;
  typedef ublas::vector<double> DVector;
  typedef ublas::zero_vector<double> DVector_zero;
  typedef ublas::scalar_vector<double> DVector_scalar;
  typedef ublas::matrix<int> IMatrix;
  typedef ublas::zero_matrix<int> IMatrix_zero;
  typedef ublas::matrix<double> DMatrix;
  typedef ublas::zero_matrix<double> DMatrix_zero;
  typedef ublas::identity_matrix<double> DMatrix_identity;

  typedef ublas::coordinate_matrix<double, ublas::column_major, 0> SparseMatrix;
  typedef SparseMatrix::const_iterator1 row_iter_t;
  typedef SparseMatrix::const_iterator2 col_iter_t;

}
#endif
