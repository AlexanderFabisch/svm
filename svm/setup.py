from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
import numpy
import os


def strict_prototypes_workaround():
    # Workaround to remove '-Wstrict-prototypes' from compiler invocation
    (opt,) = get_config_vars('OPT')
    os.environ['OPT'] = " ".join(
        flag for flag in opt.split() if flag != '-Wstrict-prototypes'
    )


if __name__ == '__main__':
    strict_prototypes_workaround()

    extensions = [
        Extension(
            "_svm",
            ["_svm.pyx",],
            include_dirs = [numpy.get_include(),],
            define_macros = [("NDEBUG",),],
            extra_compile_args = [
                "-std=c++11",
                # disable warnings caused by Cython using the deprecated
                # NumPy C-API
                "-Wno-cpp", "-Wno-unused-function",
                "-O3"
            ],
            language = "c++"
        )
    ]
    setup(ext_modules=cythonize(extensions))