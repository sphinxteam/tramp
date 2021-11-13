Installation
------------

.. note::
    Tree-AMP package requires the following packages

    - python>=3.6
    - numpy/pandas/scipy/matplotlib
    - networkx==1.11
    - daft

    Currently the package does not support networkx 2.xx and will throw errors.
    We plan to upgrade to networkx 2.xx at some point.


To install the package, go to the folder where setup.py is located and run:

.. code-block::

    pip install .


If you want to install in development mode (changes to the repository will immediately affect the installed package without needing to re-install):

.. code-block::

    pip install --editable .

To install the package on a remote machine directly from the github repo:

.. code-block::

    pip install git+https://github.com/sphinxteam/tramp.git


See `installing from sources <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-from-source>`_ for more details.
In all three cases, the necessary requirements should be automatically installed.
