'''Wrappers around Python logging routines for a parallel processing setup'''

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

import logging
import cPickle
from collections import defaultdict

import mpihelper

# Define levels for results logger
ITERATION = logging.DEBUG
FINAL = logging.INFO
REPLICATE = logging.WARNING

# Define tag used when sending result log records using MPI
RESULT_TAG = 42

def init(level):
    '''Initialize the global logger and enable it for the given level.

    Arguments:
    level -- logging level for the global logger
    '''
    getLogger().setLevel(level)
    getLogger().addHandler(NullHandler())

def getLogger(name = 'unyque'):
    '''Return the logger associated with the given name. The main logger is
    returned by default.

    Arguments:
    name -- Name of the logger
    '''
    return logging.getLogger(name)

def addHandler(handler):
    '''Add a new handler to the main logger. The handler is added only if this
    is the master processor, since we don't want workers to log anything.

    Arguments:
    handler -- Handler object to be added
    '''

    if mpihelper.rank == mpihelper.MASTER:
        getLogger().addHandler(handler)

def init_results_logger(filename, level):
    '''Initialize the results logger to store results corresponding to the
    given level and higher in the specified file.

    Arguments:
    filename -- Relative path of the file into which results can be appended
    level -- Logging level that is used to filter results
    '''

    logger = getLogger('unyque.results')
    logger.read_results(filename)
    logger.addHandler(ResultsHandler(filename))
    logger.setLevel(level)

class NullHandler(logging.Handler):
    '''Default handler that is always added to the main logger and is used to
    consume any log messages without throwing up warnings about no handlers
    being found.
    '''

    def emit(self, record):
        '''Do nothing when a record is received.
        '''
        pass

class ResultsLogger(logging.Logger):
    '''Custom logger that is useful when computing and storing results. It is
    designed for efficient operation in a multiprocessing environment.
    '''

    def __init__(self, name):
        '''Default constructor for ResultsLogger.
        '''
        logging.Logger.__init__(self, name)
        self.completed_results = None

    def read_results(self, filename):
        '''Read results from the file corresponding to the given filename and
        store their IDs in the cache. The master processor reads and processes
        the file and broadcasts the information to all the workers.
        '''

        if mpihelper.rank == mpihelper.MASTER:
            # Read from results log if it exists
            results = list()
            try:
                with open(filename, 'rb') as f:
                    while True:
                        try:
                            results.append(cPickle.load(f))
                        except EOFError:
                            break
            except IOError as ioe:
                if 'No such file' in str(ioe): # File does not exist
                    pass
                else: # Unexpected error, so re-raise
                    raise ioe

            # Gather lists of completed results corresponding to each tag.
            # If result dictionary contains 'tag' and 'replicate' keys, then
            # add its replicate ID to the list, else ignore the result
            self.completed_results = defaultdict(list)
            for r in results:
                if 'tag' in r and 'replicate' in r:
                    self.completed_results[r['tag']].append(r['replicate'])

            # Send this dictionary to workers
            mpihelper.comm.bcast(self.completed_results,
                                 root = mpihelper.MASTER)
        else:
            # Obtain completed results from master
            self.completed_results = mpihelper.comm.bcast(
                root = mpihelper.MASTER)


    def isEnabledFor(self, level, resultID = None):
        '''Overloaded function that checks whether log records at a given level
        will be handled by this logger. It also takes an optional resultID
        argument to determine whether a particular result is to be computed.

        Arguments:
        resultID -- Tuple with tag and index of result to be computed.
        '''

        # Return True if logger is enabled for this level and if either the
        # resultID has not been specified or if the result has not been computed
        # so far
        return logging.Logger.isEnabledFor(self, level) and \
            (not resultID or
             resultID[1] not in self.completed_results[resultID[0]])

    def listen_for_result(self):
        '''If this is the master processor and there are workers, then listen
        for an incoming result log record from one of the workers.
        '''
        if mpihelper.rank == mpihelper.MASTER and mpihelper.size > 1:

            status = mpihelper.mpi.Status()

            # Wait until one of the workers completes its task
            mpihelper.comm.Probe(mpihelper.mpi.ANY_SOURCE, tag = RESULT_TAG,
                                 status = status)

            # Determine the rank of the worker and get the result
            waiting_worker = status.Get_source()
            result = mpihelper.comm.recv(source = waiting_worker,
                                         tag = RESULT_TAG)

            # Post result to the log
            self.handle(result)

class ResultsHandler(logging.Handler):
    '''Handler used to store results posted to the results logger, by writing
    the pickled results to a file. When initialized on worker processors, it
    merely sends the result to the master processor using MPI, while the
    corresponding object initialized on the master processor performs the task
    of actually writing the results.
    '''

    def __init__(self, filename):
        '''Constructor for creating a ResultsHandler object that writes
        results posted in the form of log records, to the given file.

        Arguments:
        filename -- String specifying file location where results can be saved
        '''

        logging.Handler.__init__(self)
        self.filename = None
        if mpihelper.rank == mpihelper.MASTER:
            self.filename = filename

    def emit(self, record):
        '''If this is the master processor, save the given record by pickling
        its attribute dictionary and appending the pickled form to the file. If
        this is the worker, then send the record to the master.

        Arguments:
        record -- the LogRecord instance that is to be handled
        '''

        if mpihelper.rank == mpihelper.MASTER:
            with open(self.filename, 'ab') as f:
                cPickle.dump(record.__dict__, f)
        else:
            mpihelper.comm.send(record, dest = mpihelper.MASTER,
                                tag = RESULT_TAG)

# Set the logger class to our custom results logger
logging.setLoggerClass(ResultsLogger)
