# Building docs

We use Sphinx for generating the reference documentation for Tree-AMP.

## Instructions

In addition to installing Tree-AMP and its dependencies, install the Python
packages need to build the documentation by entering:

    pip install -r requirements.txt

in the ``docs/`` directory.

To build the HTML documentation, enter:

    make html

in the ``docs/`` directory. If all goes well, this will generate a
``_build/0.1/html/`` subdirectory containing the built documentation.
