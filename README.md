Support Vector Machine (SVM)
============================

This is a binary SVM and is trained using the SMO algorithm.

* Reference: [The Simplified SMO Algorithm](
  http://math.unt.edu/~hsp0009/smo.pdf)
* Based on Karpathy's [svm.js](https://github.com/karpathy/svmjs)

This implementation is based on Cython, NumPy, and scikit-learn.

Installation
------------

The packages Cython, numpy and scikit-learn are required. You
can clone this repository and run the installation script with

    sudo python setup.py install

Tests
-----

In order to run the unit tests, you must build the Cython extension
first

    python setup.py build_ext --inplace

and then run nosetests.
