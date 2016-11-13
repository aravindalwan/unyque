'''Uncertainty propagation routines based on stochastic collocation'''

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

import pickle

from . import logmanager
from .rdomain import RandomDomain
from . import mpihelper

class StochasticCollocation(object):
    '''Basic solver that uses stochastic collocation to propagate uncertainties
    '''

    # Initialize loggers used in this class
    _log = logmanager.getLogger('unyque.scol')
    _results_log = logmanager.getLogger('unyque.results')

    def __init__(self, fevaluator, smodel, bounds):
        '''Constructor for StochasticCollocation object.

        Arguments:
        fevaluator -- FunctionEvaluator object used for evaluating function
        smodel -- Stochastic model object used for evaluating PDF
        bounds -- List of tuples of the form (upper bound, lower bound)
        corresponding to each random variable in random_vars
        '''

        self._log.info('Initializing stochastic collocation framework')
        self.function_evaluator = fevaluator
        RandomDomain.set_number_of_dimensions(smodel.ndim)
        self.domain = RandomDomain(smodel, bounds)

    def propagate(self, tol = 1.0e-5, min_order = 2, max_order = 15):
        '''Construct the stochastic collocation interpolant, propagate the
        stochastic model through it and compute statistics. The interpolant is
        successively refined by increasing its order of interpolation until the
        error indicator for interpolation error meets a tolerance condition.

        Arguments:
        tol -- Threshold value used in tolerance condition on error indicator
        min_order -- Minimum order of interpolation
        max_order -- Maximum value to which the interpolation order is increased
        '''

        if max_order < min_order:
            raise ValueError('Value of max_order cannot be less than that of ' +
                             'min_order.')

        q = min_order

        for q in range(min_order, max_order+1):

            self._log.info('Constructing interpolant of order %d', q)
            RandomDomain.set_interpolant_order(q)

            self._log.debug('Updating hierarchical surpluses')
            self.domain.update_hierarchical_surpluses(
                self.function_evaluator(self.domain.compute_grid()))

            self._log.debug('Computing error indicator')
            error_indicator = self.domain.compute_error_indicator()

            # Log the result obtained from the current iteration
            self._compute_result(logmanager.ITERATION,
                                 'Generating results for order {0}'.format(q))

            # Break out of loop if tolerance condition is met
            if error_indicator < tol:
                break

        if q == max_order:
            self._log.warning(
                'Interpolation order is at maximum value, so it is possible ' +
                'that error tolerance condition has not yet been satisfied.')

        # Log the final result
        self._compute_result(logmanager.FINAL, 'Generating final results')

    def _compute_result(self, level, log_message):
        '''Compute statistics and post the result into the ResultLogger with
        the given level and message.
        '''
        if self._results_log.isEnabledFor(level):
            self._log.log(level, log_message)
            moments = self.domain.compute_moments([1,2])
            mean = moments[0]
            var = moments[1] - mean*mean
            result = {'order': self.domain.q, 'mean': mean, 'var': var,
                      'eind': self.domain.compute_error_indicator(),
                      'nfeval': self.function_evaluator.count}
            self._results_log.log(
                level, 'Order: %(order)2d     Mean: %(mean)1.10e     ' +
                'Variance: %(var)1.10e     Err. Ind.: %(eind)1.7e     ' +
                'Func. Eval.: %(nfeval)6d', result, extra = result)

    def set_stochastic_model(self, smodel):
        '''Replace existing stochatic model with a new one.

        Arguments:
        smodel -- New stochastic model object
        '''
        self.domain.stochastic_model = smodel

    def propagate_replicate(self, rep, tag = ''):
        '''Perform UQ for the new stochastic model replicate. It is assumed that
        the stochastic model has been already set to the new replicate.

        Arguments:
        rep -- index of the replicate to be processed
        tag -- string used to identify the result at a later stage
        '''

        # This method is meant to be called on each replicate on every processor.
        # We first check to see if the given replicate is meant to be processed
        # by this processor. All replicates are processed by the master, if there
        # is only one processor and by the first worker, if there are two. In
        # case of multiple workers, the replicates are split evenly among them.

        # Check if this replicate should be processed at all
        if self._results_log.isEnabledFor(logmanager.REPLICATE, (tag, rep)):

            # Compute my rank among the workers
            worker_rank = mpihelper.rank if \
                mpihelper.rank < mpihelper.MASTER else mpihelper.rank - 1

            # Perform the computation if there is only one processor or if this
            # is the worker processor that is assigned this computation
            if mpihelper.size == 1 or (
                mpihelper.rank != mpihelper.MASTER and \
                    rep % (mpihelper.size - 1) == worker_rank):

                # Compute statistics and log the result
                moments = self.domain.compute_moments([1,2])
                mean = moments[0]
                var = moments[1] - mean*mean
                result = {'replicate': rep, 'mean': mean, 'var': var,
                          'rank': mpihelper.rank, 'tag': tag}
                self._results_log.log(
                    logmanager.REPLICATE, 'Replicate: %(replicate)3d     ' +
                    'Mean: %(mean)1.10e     Variance: %(var)1.10e     ' +
                    'Tag: %(tag)s', result, extra = result)

            else: # Listen for a processed result from one of the workers

                self._results_log.listen_for_result()

    def propagate_sample_set(self, rep, sample_set, tag = ''):
        '''Propagate samples in the given sample set replicate. Each set is a
        list of tuples corresponding to points in the random domain.

        Arguments:
        rep -- index of the replicate to be processed
        sample_set -- points in the random domain given as a list of tuples
        tag -- string used to identify the result at a later stage
        '''

        # This method is meant to be called on each replicate on every processor.
        # We first check to see if the given replicate is meant to be processed
        # by this processor. All replicates are processed by the master, if there
        # is only one processor and by the first worker, if there are two. In
        # case of multiple workers, the replicates are split evenly among them.

        # Check if this replicate should be processed at all
        if self._results_log.isEnabledFor(logmanager.REPLICATE, (tag, rep)):

            # Compute my rank among the workers
            worker_rank = mpihelper.rank if \
                mpihelper.rank < mpihelper.MASTER else mpihelper.rank - 1

            # Perform the computation if there is only one processor or if this
            # is the worker processor that is assigned this computation
            if mpihelper.size == 1 or (
                mpihelper.rank != mpihelper.MASTER and \
                    rep % (mpihelper.size - 1) == worker_rank):

                # Evaluate interpolant for each sample in the sample set
                result_set = self.domain.interpolate(sample_set)

                # Compute empirical statistics and log the result
                m = len(sample_set)
                mean = sum(result_set)/m
                var = (sum(map(lambda x: x**2, result_set))-m*mean**2)/(m-1)
                result = {'replicate': rep, 'mean': mean, 'var': var,
                          'tag': tag}
                self._results_log.log(
                    logmanager.REPLICATE, 'Replicate: %(replicate)3d     ' +
                    'Mean: %(mean)1.10e     Variance: %(var)1.10e     '
                    'Tag: %(tag)s', result, extra = result)

            else: # Listen for a processed result from one of the workers

                self._results_log.listen_for_result()
