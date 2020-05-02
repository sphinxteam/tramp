Getting started
===============

Get started with our package with these steps:

Requirements
------------
.. tip::
    - python>=3.6
    - numpy/pandas/scipy/matplotlib
    - networkx==1.11
    - daft 
..


.. warning::
    Currently the package does not support networkx 2.xx and will throw unexpected errors. We plan to upgrade to networkx 2.xx at some point.
..



Installation
------------
 

To install the package, go to the folder where setup.py is located and run:

.. code-block:: console

    pip install .
    
or if you want to install in development mode (changes to the repository will immediately affect the installed package without needing to re-install):

.. code-block:: console

    pip install --editable .

To install the package on a remote machine directly from the github repo:

.. code-block:: console
    
    pip install git+https://github.com/sphinxteam/tramp.git

See `installing from sources <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-from-source>`_ for more details. In both cases, the necessary requirements should be automatically installed.


Citation
--------