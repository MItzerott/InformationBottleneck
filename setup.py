from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

'''
extensions = [
    Extension("kl_divergence", ["kl_divergence.pyx"],
        include_dirs=[...],
        libraries=[...],
        library_dirs=[...]),
    Extension("entropy", ["entropy.pyx"],
        include_dirs=[...],
        libraries=[...],
        library_dirs=[...]),
    # Everything but primes.pyx is included here.
    Extension("*", ["*.pyx"],
        include_dirs=[...],
        libraries=[...],
        library_dirs=[...]),
]

'''
extensions = [
    Extension(
        '*', ["*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name = 'KL_Divergence',
    ext_modules = cythonize(extensions)
)

setup(
    name = 'Entropy',
    ext_modules = cythonize(extensions)
)

#execute with python setup.py build_ext --inplace