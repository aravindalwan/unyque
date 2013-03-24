'''Stochastic model for a random variable'''

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

import numpy as np
import itertools as it
import operator as op
from scipy.special import erf
from scipy.stats.distributions import norm

class KernelDensityModel(object):
    '''Multivariate stochastic model generated using kernel density estimation.
    The joint PDF is represented as a weighted sum of Gaussian prototypes
    centered at a set of points in the random space. The Gaussian prototypes are
    assumed to have diagonal covariance matrices.
    '''

    def __init__(self, locations, bandwidths, weights):
        self.set_parameters(locations, bandwidths, weights)

    def set_parameters(self, locations, bandwidths, weights):
        '''Convenience method to set parameters of the kernel density estimate.

        Arguments:
        locations -- (m,d)-array representing points in the random space
        bandwidths -- (m,d)-array representing bandwidths of each prototype
        weights -- (m,1)-array representing weighting values for each prototype
        '''

        self.m, self.ndim = np.shape(locations)
        if not ((bandwidths.shape == (self.m, self.ndim)) and
                (weights.shape == (self.m, ))):
            raise ValueError(
                'The attributes locations, bandwidths and weights must be 2D ' +
                'arrays with shape (m, d), (m, d) and (m, ) respectively.')

        self.locations = locations
        self.bandwidths = bandwidths
        self.weights = weights

    def __call__(self, eval_points):
        '''Evaluates the PDF at the given points in the random space

        Arguments:
        eval_points -- (n,d)-array representing coordinates in random space
        '''

        if not (len(np.shape(eval_points)) == 2 and
                np.shape(eval_points)[1] == self.ndim):
            raise ValueError(
                'The attribute eval_points must be a 2D array with shape ' +
                '(n, {d}), where n is the number of evaluation points'.format(
                    d = self.ndim))

        npoints = np.shape(eval_points)[0]
        result = np.zeros(npoints,)
        sum_weights = np.sum(self.weights)

        if npoints >= self.m:
            # loop over locations
            for i in xrange(self.m):
                result += self.weights[i]*np.prod(norm.pdf(
                        eval_points, self.locations[i], self.bandwidths[i]),
                                                  axis = 1)/sum_weights
        else:
            # loop over evaluation points
            for i in xrange(npoints):
                result[i] = np.sum(self.weights*np.prod(norm.pdf(
                            eval_points[i], self.locations, self.bandwidths),
                                                        axis = 1)/sum_weights)
        return result

    def expectation(self, basis_function):
        '''Returns the expectation of the given basis function with respect to
        this PDF. The basis function is represented as a sequence of functions
        along each dimension. Each univariate function is expressed as a tuple
        comprising the upper and lower limits as well as its level.

        Arguments:
        basis_function -- List of tuples as described above.
        '''

        kernel = np.ones(self.m,)
        for dim, univariate_function in enumerate(basis_function):
            kernel *= self._univariate_integral(dim, *univariate_function)

        return np.sum(self.weights*kernel)/np.sum(self.weights)

    def _univariate_integral(self, dim, lb, ub, pt):
        '''Returns the integral of a piece-wise linear basis function between
        lb and ub, with gaussian kernels along the given dimension. The form of
        the basis function is determined by the pt argument, which is a tuple
        comprising the level and index of the gridpoint corresponding to the
        basis function.
        '''

        mu = self.locations[:,dim]
        sigma = self.bandwidths[:,dim]
        level, idx = pt
        value = np.zeros(self.m,)

        # Loop over all parts of the piece-wise linear interpolant
        for i in range(2):

            # Compute the integration limits for the current piece
            if level <= 2:
                if i > 0:
                    break # Only one part for levels 1 and 2
                lo, hi = lb, ub
            else:
                (lo, hi) = (lb, (lb+ub)/2) if i == 0 else ((lb+ub)/2, ub)

            # Determine the form of the basis function in this region
            if level == 1:
                a, b = 1., 0.
            elif level == 2:
                a = 0.5*(1 + (-1.)**idx*(hi+lo)/(hi-lo))
                b = (-1.)**(idx+1)/(hi-lo)
            else:
                a = 0.5*(1 - (-1.)**i*(hi+lo)/(hi-lo))
                b = (-1.)**i/(hi-lo)

            # Integrate function with respect to kernels along given dimension
            dPhi = 0.5*(erf((hi - mu)/sigma/np.sqrt(2)) -
                        erf((lo - mu)/sigma/np.sqrt(2)))
            dphi = norm.pdf((hi - mu)/sigma) - norm.pdf((lo - mu)/sigma)
            value += (a+b*mu)*dPhi - sigma*b*dphi

        return value

class IndependentVariablesModel(object):
    '''Multivariate stochastic model that describes the joint PDF of two or more
    random variables that are mutually independent. The joint PDF is the product
    of their marginal PDFs.
    '''

    def __init__(self, random_vars):
        '''Constructor for the IndependentVariablesModel class, that creates an
        object representing the joint model using a list of random variables.

        Arguments:
        random_vars -- List of random variables as MarginalPDF objects
        '''
        self._random_variables = random_vars
        self.ndim = len(random_vars)

    def __call__(self, eval_points):
        '''Evaluates the PDF at the given points in the random space

        Arguments:
        eval_points -- (n,d)-array representing coordinates in random space
        '''

        if not (len(np.shape(eval_points)) == 2 and
                np.shape(eval_points)[1] == self.ndim):
            raise ValueError(
                'The attribute eval_points must be a 2D array with shape ' +
                '(n, {d}), where n is the number of evaluation points'.format(
                    d = self.ndim))
        return reduce(op.mul, (pdf(eval_points[:,dim]) for dim, pdf in
                               enumerate(self._random_variables)) )

    def expectation(self, basis_function):
        '''Returns the expectation of the given basis function with respect to
        this PDF. The basis function is represented as a sequence of functions
        along each dimension. Each univariate function is expressed as a tuple
        comprising the upper and lower limits as well as its level.

        Arguments:
        basis_function -- List of tuples as described above.
        '''
        return reduce(op.mul, (self._random_variables[dim].expectation(
                    *f1d) for dim, f1d in enumerate(basis_function)) )

class MarginalPDF(object):
    '''Base marginal PDF object that describes the PDF of a random variable
    and performs functions related to evaluation and integration of the PDF.
    '''

    def __init__(self):

        # Set support of PDF to be infinite in both directions
        self._min = -float('inf')
        self._max = float('inf')

    def __call__(self, x):
        '''Evaluates the PDF at the given point, x
        '''
        return (x >= self._min)*(x <= self._max)/(self._max - self._min)

    def expectation(self, lb, ub, pt):
        '''Returns the expectation of a piece-wise linear basis function between
        lb and ub, with this marginal PDF. The form of the basis function is
        determined by the pt argument, which is a tuple comprising the level and
        index of the gridpoint corresponding to the basis function.
        '''

        level, idx = pt
        value = 0.0

        # Loop over all parts of the piece-wise linear interpolant
        for i in range(2):

            # Compute the integration limits for the current piece
            if level <= 2:
                if i > 0:
                    break # Only one part for levels 1 and 2
                lo, hi = lb, ub
            else:
                (lo, hi) = (lb, (lb+ub)/2) if i == 0 else ((lb+ub)/2, ub)

            # Determine the form of the basis function in this region
            if level == 1:
                a, b = 1., 0.
            elif level == 2:
                a = 0.5*(1 + (-1.)**idx*(hi+lo)/(hi-lo))
                b = (-1.)**(idx+1)/(hi-lo)
            else:
                a = 0.5*(1 - (-1.)**i*(hi+lo)/(hi-lo))
                b = (-1.)**i/(hi-lo)

            # Integrate function with respect to kernels along given dimension
            value += self._univariate_integral(a, b, lo, hi)

        return value

    def _univariate_integral(self, a, b, lo, hi):
        '''Evaluates the integral of the linear function, a + b*x, with respect
        to the PDF. The function has a finite support characterized by its
        bounds, lo and hi.
        '''

        # Restrict integration range to the support of the PDF
        lo = max(lo, self._min)
        hi = min(hi, self._max)

        # Compute integral
        value = a*(hi - lo)/(self._max-self._min) + \
            b*(hi**2/2 - lo**2/2)/(self._max-self._min)

        return value

class UniformPDF(MarginalPDF):
    '''Marginal PDF of a random variable that varies uniformly between range_min
    and range_max.
    '''

    def __init__(self, range_min, range_max):
        super( UniformPDF, self ).__init__()

        if range_max <= range_min:
            raise ValueError(
                'Value of range_max has to be greater than range_min')

        self._min = float(range_min)
        self._max = float(range_max)

class UserDefinedPDF(MarginalPDF):
    '''Marginal PDF of a random variable that is described by a user-defined
    probability density function.
    '''
    def __init__(self, pdf, range_min, range_max):
        super( UserDefinedPDF, self ).__init__()

        self._pdf = pdf

        if range_max <= range_min:
            raise ValueError(
                'Value of range_max has to be greater than range_min')

        self._min = float(range_min)
        self._max = float(range_max)

    def __call__(self, x):
        return self._pdf(x)

class KernelDensityPDF(MarginalPDF):
    '''Marginal PDF obtained using Kernel Density Estimation, where the PDF
    is written as a weighted sum of contributions from Gaussian kernels centered
    at each data point.
    '''

    def __init__(self, locations, bandwidths, weights):

        super( KernelDensityPDF, self ).__init__()
        self.set_parameters(locations, bandwidths, weights)

    def set_parameters(self, locations, bandwidths, weights):
        '''Convenience method to set parameters of the kernel density estimate.
        '''

        if not ((locations.shape == bandwidths.shape == weights.shape) and
                (len(locations.shape) == 1)):
            raise ValueError(
                'The attributes locations, bandwidths and weights must be 1D ' +
                'arrays with the same number of elements')

        self.locations = locations
        self.bandwidths = bandwidths
        self.weights = weights
        self.m = len(self.locations)

    def __call__(self, eval_points):

        npoints = np.size(eval_points)
        y = np.atleast_1d(eval_points).flatten()
        result = np.zeros_like(y)
        sum_weights = np.sum(self.weights)

        if npoints >= self.m:
            # loop over locations
            for i in xrange(self.m):
                result += self.weights[i]*norm.pdf(
                    y, self.locations[i], self.bandwidths[i])/sum_weights
        else:
            # loop over evaluation points
            for i in xrange(npoints):
                result[i] = np.sum(self.weights*norm.pdf(
                        y[i], self.locations, self.bandwidths))/sum_weights

        result = np.reshape(result, np.shape(eval_points))

        return result

    def _univariate_integral(self, a, b, lo, hi):
        '''Evaluates the integral of the linear function, a + b*x, with respect
        to the PDF. The function has a finite support characterized by its
        bounds, lo and hi.
        '''

        # Restrict integration range to the support of the PDF
        lo = max(lo, self._min)
        hi = min(hi, self._max)

        # Integrate function with respect to kernels along given dimension
        dPhi = 0.5*(erf((hi - self.locations)/self.bandwidths/np.sqrt(2)) -
                    erf((lo - self.locations)/self.bandwidths/np.sqrt(2)))
        dphi = norm.pdf((hi - self.locations)/self.bandwidths) - \
            norm.pdf((lo - self.locations)/self.bandwidths)
        return np.sum(self.weights*(a+b*self.locations)*dPhi -
                      self.bandwidths*b*dphi)/np.sum(self.weights)
