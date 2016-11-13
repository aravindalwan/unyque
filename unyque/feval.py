'''Framework to evaluate a function at different points in the random space'''

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

import decimal as dec
import itertools as it
import pickle

from . import logmanager

class FunctionEvaluator(object):
    '''Class to aggregate function evaluations i.e. the system response for
    different sets of parameter values, which correspond to various points in
    the random space. The evaluations are cached to avoid redundant function
    calls. In order to meaningfully perform caching, the parameter values are
    rounded to DECIMAL_PRECISION.
    '''

    _log = logmanager.getLogger('unyque.feval')

    # Number of significant digits to retain in the number
    DECIMAL_PRECISION = 10

    # Frequency at which cache file is saved, expressed as a percentage of the
    # number of new function evaluations at each step
    CACHE_SAVE_FREQUENCY = 0.01

    # Frequency at which progress is logged during function evaluations
    LOGGING_FREQUENCY = 0.1

    def __init__(self, func, vectorize = False, use_cache = True,
                 cache_file = None):
        self._log.info(
            'Initializing function evaluator with vectorize %s, use_cache %s ' +
            'and cache_file %s', vectorize, use_cache, cache_file)
        self.solver = func
        self.vectorize = vectorize
        self.use_cache = use_cache
        self._cache_file = cache_file
        self._cache = {} if self.use_cache else None
        self._func_evaluations = 0

        if self.use_cache and self._cache_file is not None:
            try:
                with open(self._cache_file, 'rb') as pf:
                    cache = pickle.load(pf)
                    self._cache.update(cache)
                    self._func_evaluations = len(self._cache)
            except IOError as ioe:
                if 'No such file' in str(ioe):
                    # File does not exist, so create it later
                    self._log.warning(
                        'Cache file does not exist, so creating a new one')
                else: # Unknown error, so re-raise
                    raise ioe

    @property
    def count(self):
        return self._func_evaluations

    def __call__(self, parameter_sets):
        '''Return the function evaluations at the given parameter sets
        '''

        # Convert parameter sets to list of tuples of Decimal objects
        parameter_sets = self._process(parameter_sets)

        # Pick out the sets that are not in the cache already
        new_sets = [pset for pset in parameter_sets
                    if not self.use_cache or pset not in self._cache]

        # Evaluate parametric solver or function on each new parameter set
        new_results = list()
        if len(new_sets) > 0:
            if self.vectorize: # Function expects vectorized input
                new_results = self.solver(
                    [map(float, pset) for pset in new_sets])
            else:
                new_results = ((i, self.solver(*map(float, pset)))
                               for i, pset in enumerate(new_sets))

        self._func_evaluations += len(new_sets)

        # Collate results
        if self.use_cache:

            # First add new results to cache
            counter = 0
            cache_point = max(10, int(
                    self.CACHE_SAVE_FREQUENCY*len(new_sets)))
            log_point = max(10, int(self.LOGGING_FREQUENCY*len(new_sets)))
            for position, result in new_results:

                self._cache[new_sets[position]] = result
                counter += 1

                # Save to temporary cache to avoid corrupting main cache file
                if self._cache_file and counter % cache_point == 0:
                    with open(self._cache_file + '.tmp', 'wb') as pf:
                        pickle.dump(self._cache, pf)

                # Log progress information
                if counter % log_point == 0:
                    completed_fraction = float(counter)/len(new_sets)
                    done = int(100*completed_fraction)
                    remaining = 100 - done
                    self._log.debug(('[' + '='*done + ' '*remaining + '] ' +
                                    '{count} of {total} left').format(
                            count = counter, total = len(new_sets)))

            # Save into main cache file if new results were added
            # This avoids corrupting cache file if there are no updates
            if self._cache_file is not None and counter > 0:
                with open(self._cache_file, 'wb') as pf:
                    pickle.dump(self._cache, pf)

            # Return results from cache
            results = [self._cache[pset] for pset in parameter_sets]

        else:

            # No caching done, so function was evaluated at every parameter set.
            # Just return a sorted list of all the results in new_results
            results = [None]*len(new_sets)
            for position, result in new_results:
                results[position] = result

        self._log.info('Finished evaluating function for %d parameter sets, ' +
                       'of which %d are new evaluations', len(parameter_sets),
                       len(new_sets))

        return results

    @classmethod
    def _process(cls, parameter_sets):
        '''Convert values to Decimal objects, quantize them to desired precision
        and return them as a list of tuples so that they can be cached
        '''

        # Convert parameter values to Decimal objects
        decimal_sets = ((dec.Decimal('{0!r}'.format(p)) for p in pset)
                        for pset in parameter_sets)

        # Quantize Decimal objects so that they have DECIMAL_PRECISION digits.
        # p.adjusted() gives the exponent of p itself, so the precision must be
        # added to this value to give the number of digits to retain in the
        # final number
        quantized_sets = (( p.quantize(dec.Decimal('{0}'.format(
                            10**(-cls.DECIMAL_PRECISION+p.adjusted()))))
                            for p in pset ) for pset in decimal_sets)

        context = dec.getcontext()
        if context.flags[dec.Inexact]:
            warn_message = 'Rounding errors may have occurred. Please set ' \
                'feval.DECIMAL_PRECISION to a higher value to avoid seeing ' \
                'this warning.'
            cls._log.warning(warn_message)
            context.clear_flags()

        return [tuple(pset) for pset in quantized_sets]
