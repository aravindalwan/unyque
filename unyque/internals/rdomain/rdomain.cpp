#include "rdomain.hpp"

bp::list rdomain::ComputeHierarchicalSurpluses(bp::list gridLevels,
					       bp::list gridCoords,
					       bp::list funcValues,
					       int nMoments) {

  int nGrid, nDim, counter, interpLevel, levelCheck;
  bp::list hierSurpluses, listItem;
  double basisFunc;

  // Get number of grid points
  nGrid = len(gridLevels);
  assert (len(gridCoords) == nGrid);
  assert (len(funcValues) == nGrid);

  // Get number of dimensions
  listItem = bp::extract<bp::list>(gridLevels[0])();
  nDim = len(listItem);
  listItem = bp::extract<bp::list>(gridCoords[0])();
  assert (len(listItem) == nDim);

  // Initialize arrays and matrices
  ublas::matrix<int> gLevels(nGrid, nDim + 1);
  ublas::matrix<double> grid(nGrid, nDim);
  ublas::matrix<double> hSurpluses(nGrid, nMoments);
  ublas::vector<double> fValues(nGrid);
  ublas::vector<double> pli(nMoments);

  // Copy matrices from lists
  for (int i = 0; i < nGrid; i++) {

    listItem = bp::extract<bp::list>(gridLevels[i])();
    interpLevel = 0;
    for (int j = 0; j < nDim; j++) {
      gLevels(i,j) = bp::extract<int>(listItem[j])();
      interpLevel += gLevels(i, j);
    }
    gLevels(i,nDim) = interpLevel;

    listItem = bp::extract<bp::list>(gridCoords[i])();
    for (int j = 0; j < nDim; j++)
      grid(i,j) = bp::extract<double>(listItem[j])();

    fValues(i) = bp::extract<double>(funcValues[i])();

  }

  // Compute hierarchical surpluses
  for (int i = 0; i < nGrid; i++) {

    if (gLevels(i,nDim) == nDim) {

      for (int j = 0; j < nMoments; j++)
        hSurpluses(i,j) = pow(fValues(i), j+1);

    } else {

      counter = 0;
      for (int j = 0; j < nMoments; j++)
        pli[j] = 0.0;

      // Iterate over all lower order gridpoints
      while (gLevels(counter,nDim) < gLevels(i,nDim)) {
        levelCheck = 0;
        for (int j = 0; j < nDim; j++) // Check each dimension
          levelCheck += (gLevels(counter,j) > gLevels(i,j));
        if (!levelCheck) { // this gridpoint is to be included
          basisFunc = 1.0;
          for (int j = 0; j < nDim; j++)
            basisFunc *= BasisFunction1D(gLevels(counter,j), grid(i,j),
					       grid(counter,j));
          for (int j = 0; j < nMoments; j++)
            pli[j] += hSurpluses(counter,j)*basisFunc;
        }
        counter++;
      }

      // Store hierarchical surpluses
      for (int j = 0; j < nMoments; j++)
        hSurpluses(i,j) = pow(fValues(i),j+1) - pli[j];

    }

  }

  // Copy hierarchical surpluses as list of lists
  for (int i = 0; i < nGrid; i++) {
    bp::list sublist;
    for (int j = 0; j < nMoments; j++)
      sublist.append(hSurpluses(i,j));
    hierSurpluses.append(sublist);
  }

  return hierSurpluses;

}

double rdomain::BasisFunction1D(int level, double evalPoint, double gPoint) {

  int m;

  assert(level > 0); // Check that level has proper value

  if (level == 1)
    return 1.0;
  else {
    m = (1<<(level-1)) + 1; // Number of gridpoints at this level
    if (std::abs(evalPoint - gPoint) < 1.0/(m-1)) // Basis function is non-zero
      return (1.0 - (m-1)*std::abs(evalPoint - gPoint));
    else
      return 0.0;
  }

}
