#! /usr/bin/env python
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage("svm")
    return config

def setup_package():
    setup(
        name="svm",
        maintainer="Alexander Fabisch",
        maintainer_email="afabisch+github@googlemail.com",
        description="Support Vector Machine in Python",
        license="MIT",
        url="https://github.com/AlexanderFabisch/svm",
        configuration=configuration,
    )


if __name__ == "__main__":
    setup_package()
