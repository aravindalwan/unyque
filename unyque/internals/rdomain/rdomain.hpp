#ifndef __RDOMAIN_H
#define __RDOMAIN_H

#include <stdlib.h>
#include <math.h>
#include <cassert>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace bp = boost::python;
namespace ublas = boost::numeric::ublas;

namespace rdomain {

  bp::list ComputeHierarchicalSurpluses(bp::list gridLevels,
					bp::list gridCoords,
					bp::list fValues, int nMoments);
  double BasisFunction1D(int level, double evalPoint, double gPoint);

}
#endif
