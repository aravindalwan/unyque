'''Representation of a random variable used in stochastic collocation'''

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

class RandomDimension(object):
    '''Random dimension object that encapsulates the operations along one of the
    dimensions in random space, which corresponds to one of the random variables
    '''

    kmax = 0
    nodes = None

    def __init__(self, bound):
        self._bound = bound

    @classmethod
    def set_maximum_interp_level(cls, value):
        cls.kmax = value
        cls._init_nodes()

    @classmethod
    def _init_nodes(cls):
        '''Initialize nodes in a hierarchical fashion as a list of sublists,
        where each sublist contains the nodes added at the corresponding level
        '''
        cls.nodes = []
        if cls.kmax > 0:
            cls.nodes.append([0.5])
        if cls.kmax > 1:
            cls.nodes.append([0.0, 1.0])
        if cls.kmax > 2:
            for k in xrange(3, cls.kmax+1):
                cls.nodes.append([
                        (1.0 + 2.0*j)/(2**(k-1)) for j in xrange(2**(k-2))])

    def get_node(self, level, idx, normalized = False):
        '''Return the scaled coordinates of a node at the given level and index
        '''

        if normalized:
            return self.nodes[level-1][idx]
        else:
            lo = self._bound[0]
            hi = self._bound[1]
            return lo + (hi-lo)*self.nodes[level-1][idx]

    @classmethod
    def _interpolate(cls, pt1, x2):
        '''Evaluate basis function centered at pt1, at x2. pt1 has to be a
        tuple of the form (level, index) that specifies the interpolation level
        and the index of the node at that level. x2 is any float value between
        0 and 1, specifying the location where the basis function is to be
        evaluated.
        '''

        level1, idx1 = pt1
        x1 = cls.nodes[level1-1][idx1]

        if level1 == 1:
            return 1.0
        else:
            m = 2**(level1-1) + 1 # Number of nodes at this level
            return (abs(x1-x2) < 1./(m-1)) * (1. - (m-1)*abs(x1-x2))

    def interpolate(self, pt1, x):
        '''Evaluate basis function centered at pt1, at the location x. This
        method scales x to be in [0,1] and calls _interpolate to get the actual
        interpolated value
        '''

        lo = self._bound[0]
        hi = self._bound[1]
        if lo <= x <= hi:
            return self._interpolate(pt1, float(x-lo)/float(hi-lo))
        else:
            return 0.

    def get_basis_function(self, pt):
        '''Return bounds of the piece-wise linear basis function centered at pt.
        '''

        lo = self._bound[0]
        hi = self._bound[1]
        level, idx = pt

        if level == 1:
            return (lo, hi, pt)
        elif level == 2:
            lo = (lo + hi)/2 if idx == 1 else lo
            hi = (lo + hi)/2 if idx == 0 else hi
            return (lo, hi, pt)
        else:
            m = 2**(level-1) + 1 # Number of nodes at this level
            x = lo + (hi-lo)*self.nodes[level-1][idx]
            return (x-(hi-lo)/(m-1), x+(hi-lo)/(m-1), pt)

