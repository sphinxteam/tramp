# Building docs

We use Sphinx for generating the reference documentation for TRAMP.

## Instructions

In addition to installing TRAMP and its dependencies, install the Python
packages need to build the documentation by entering::

    pip install -r requirements.txt

in the ``docs/`` directory.

To build the HTML documentation, enter::

    make html

in the ``docs/`` directory. If all goes well, this will generate a
``build/0.1/html`` subdirectory containing the built documentation.
