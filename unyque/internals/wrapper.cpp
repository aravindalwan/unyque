#include <boost/python.hpp>

#include "rdomain.hpp"

using namespace boost::python;

BOOST_PYTHON_MODULE(_internals)
{

  def("ComputeHierarchicalSurpluses", rdomain::ComputeHierarchicalSurpluses);

}
