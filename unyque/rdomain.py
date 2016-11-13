'''Space of random variables over which stochastic collocation is performed'''

__copyright__ = 'Copyright (C) 2011 Aravind Alwan'

__license__ = '''
This file is part of UnyQuE.

UnyQuE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

UnyQuE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

For a copy of the GNU General Public License, please see
<http://www.gnu.org/licenses/>.
'''
import itertools as it
import operator as op

from .rdimension import RandomDimension
import unyque._internals as internals
from . import logmanager

class RandomDomain(object):
    '''Basic random domain object that consists of a set of random dimensions.
    '''

    _log = logmanager.getLogger('unyque.rdom')

    ndim = 0 # Number of random dimensions
    q = -1 # Interpolation level used in stochastic collocation
    _grid_indices = None # Grid points used in stochastic collocation
    NUM_MOMENTS = 2 # Number of moments to calculate

    def __init__(self, smodel, bounds):

        self._log.info('Initializing %d-dimensional random domain',
                       smodel.ndim)

        if self.ndim != smodel.ndim:
            raise StateError(
                'Dimensionality of smodel is different from RandomDomain.ndim '+
                'attribute. Set number of dimensions using ' +
                'RandomDomain.set_number_of_dimensions() before creating any ' +
                'RandomDomain object.')

        if len(bounds) != smodel.ndim:
            raise ValueError(
                'Number of tuples in bounds must equal the dimensionality of ' +
                'smodel')

        self.bounds = bounds
        self.dims = [RandomDimension(bnd) for bnd in bounds]
        self.stochastic_model = smodel
        self.hsurpluses = None

    @classmethod
    def set_number_of_dimensions(cls, value):
        cls.ndim = value

    @classmethod
    def set_interpolant_order(cls, value):
        cls.q = value

        # Maximum level of 1D interpolant will be q+1
        RandomDimension.set_maximum_interp_level(cls.q+1)

        # Initialize grid
        cls._init_grid()

        cls._log.info('Initialized grid with %d nodes', len(cls._grid_indices))

    @classmethod
    def _init_grid(cls):
        '''Initialize the grid points using the Smolyak algorithm. The actual
        node locations are not computed. For each grid point, we only store the
        indices of the corresponding 1D nodal set for each dimension as a
        tuple of the 1D grid level and the index of the point at that level.
        '''
        cls._grid_indices = list()

        # Store indices of 1D nodes at each level
        node_idx_1d = [range(len(i)) for i in RandomDimension.nodes]

        # Generate combos as list of sublists, with each sublist containing the
        # 1D interpolation levels for that combination
        combos = it.chain( *(cls._get_index_set(sumk, cls.ndim)
                             for sumk in xrange(cls.ndim, cls.ndim + cls.q + 1)) )

        # Generate indices for each gridpoint
        gridpoints = it.chain( *(it.product( *(
                        # Product of level and indices of 1D nodes
                        it.product([ki], node_idx_1d[ki-1]) for ki in combo
                        ) ) for combo in combos) )

        cls._grid_indices.extend(gridpoints)

    @staticmethod
    def _get_index_set(sumk, d):
        '''Return the index set containing 1D interpolation level combinations
        that sum up to sumk. The equivalent combinatorial problem is one of
        placing d-1 separators in the sumk-1 spaces between sumk values.
        Idea: http://wordaligned.org/articles/partitioning-with-python
        '''
        splits = it.combinations(range(1,sumk),d-1)
        return (list(map(op.sub, it.chain(s, [sumk]), it.chain([0], s)))
                for s in splits)

    def compute_grid(self):
        '''Return the grid points for this domain
        '''
        return [ [self.dims[dim].get_node(*node) for dim, node in enumerate(gp)]
                 for gp in self._grid_indices ]

    def update_hierarchical_surpluses(self, fValues):
        '''Update the hierarchical supluses using the list of function
        evaluations given in fValues
        '''

        if len(fValues) != len(self._grid_indices):
            raise ValueError('Number of function evaluations given does not ' +
                             'match the number of grid points.')

        gridLevels = [[gi[0] for gi in g] for g in self._grid_indices]
        gridCoords = [[self.dims[dim].get_node(*node, normalized = True)
                       for dim, node in enumerate(g)]
                      for g in self._grid_indices]

        self.hsurpluses = internals.ComputeHierarchicalSurpluses(
            gridLevels, gridCoords, fValues, self.NUM_MOMENTS)

    def interpolate(self, eval_points):
        '''Evaluate the interpolant at the points given as a list of tuples,
        eval_points, where each tuple contains the coordinates of a point in
        the random space
        '''

        if len(self.hsurpluses) != len(self._grid_indices):
            raise StateError(
                'Hierarchical surpluses are missing or outdated.' +
                'Call the update_hierarchical_surpluses() method first.')

        interp = list()

        for pt in eval_points:

            # Evaluate basis function as product of unvariate basis functions
            basis_funcs = [ reduce(op.mul, (self.dims[dim].interpolate(
                            node, pt[dim]) for dim, node in enumerate(gp)))
                            for gp in self._grid_indices ]

            # Interpolant is sum of basis functions weighted by hierarchical
            # surpluses
            interp.append(sum(it.imap(
                        op.mul, (h[0] for h in self.hsurpluses), basis_funcs)))

        return interp

    def compute_moments(self, moments):
        '''Compute the moments of the interpolant with respect to the given
        stochastic model
        '''

        for m in moments:
            if (m < 1) or (m > self.NUM_MOMENTS):
                raise ValueError(
                    ('Cannot compute moment of order {moment}. RandomDomain ' +
                     'only computes moments up to order {maximum}. For higher '+
                     'moments set RandomDomain.NUM_MOMENTS appropriately and ' +
                     're-run update_hierarchical_surpluses().').format(
                        moment = str(m), maximum = str(self.NUM_MOMENTS)))

        if ((not self.hsurpluses) or
            (len(self.hsurpluses) != len(self._grid_indices))):
            raise StateError(
                'Hierarchical surpluses are missing or outdated. ' +
                'Call the update_hierarchical_surpluses() method first.')

        # Evaluate expectation of basis function corresponding to each gridpoint
        # where the multivariate basis function is expressed as a product of
        # univariate basis functions
        basis_func_integrals = [ self.stochastic_model.expectation(
                [self.dims[dim].get_basis_function(node)
                 for dim, node in enumerate(gp)] )
                                 for gp in self._grid_indices ]

        # Moment is the sum of the basis function expectations weighted by
        # hierarchical surpluses
        return [sum(it.imap(op.mul, (h[m-1] for h in self.hsurpluses),
                basis_func_integrals)) for m in moments]

    def compute_error_indicator(self, modified = False):
        '''Compute the error indicator for this domain, which is the maximum
        value among the hierarchical surpluses of the gridpoints added at the
        highest level. If modified is True, then the same is computed for the
        products of the hierarchical surpluses with the expectations of the
        corresponding basis functions
        '''

        # Global interpolation level of each gridpoint
        interp_level = [ reduce(op.add, (gi[0] for gi in g))
                         for g in self._grid_indices ]

        if modified:
            return max( (self.hsurpluses[i][0]*
                         reduce(op.mul, (self.dims[dim].integrate(node)
                                         for dim, node in enumerate(gp)))
                         for i, gp in enumerate(self._grid_indices)
                         if interp_level[i] == (self.q + self.ndim)) )
        else:
            return max( (self.hsurpluses[i][0]
                         for i in xrange(len(self._grid_indices))
                         if interp_level[i] == (self.q + self.ndim)) )

class StateError(Exception):
    '''Exception that is raised when a method is called on the random domain
    object when it is not in the right state. A typical example is when one
    tries to evaluate the interpolant or compute the moments before the
    hierarchical surpluses have been updated
    '''

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
