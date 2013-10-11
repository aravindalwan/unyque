# import distribute_setup
# distribute_setup.use_setuptools()

from setuptools import setup, Extension
import sys, os
import pkg_resources

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()

version = '0.1'

install_requires = [
    # List your project dependencies here.
    # For more details, see:
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
    ]

setup(name='unyque',
      version=version,
      description='Uncertainty Quantification Environment for modeling' + \
          ' uncertainties in engineering systems',
      long_description=README + '\n\n' + NEWS,
      classifiers=[
        # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        ],
      keywords='uncertainty estimation FEM',
      author='Aravind Alwan',
      author_email='aalwan2@illinois.edu',
      url='',
      license='GPLv3',
      packages=['unyque'],
      ext_modules = [
        Extension('unyque._internals', [
                'unyque/internals/rdomain/rdomain.cpp',
                'unyque/internals/wrapper.cpp',
                ],
                  include_dirs = [
                'unyque/internals/util',
                'unyque/internals/rdomain',
                ],
                  libraries=[
                ],
                  depends = [
                'unyque/internals/util/ublas.hpp',
                'unyque/internals/rdomain/rdomain.hpp',
                ],
                  extra_compile_args = ['-Wno-unused-function'],
                  ),
        ],
      zip_safe=False,
      install_requires=install_requires,
      entry_points={
        'console_scripts':
            []
        }
      )
