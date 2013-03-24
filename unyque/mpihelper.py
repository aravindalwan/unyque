'''Wrapper around MPI4PY routines used in a parallel processing setup'''

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

import warnings

# Rank of the processor that is designated as the master processor
MASTER = 0

# Define global variables assuming single processor setup
mpi = None
comm = None
rank = 0
size = 1

# Try importing mpi4py module
try:

    from mpi4py import MPI

except ImportError: # Module does not exist - revert to single processor setup

    warnings.warn('Error importing the mpi4py module. Trying to run as a single ' +
               'processor machine instead.')

else: # Sucess! Populate global variables with appropriate values

    mpi = MPI
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
