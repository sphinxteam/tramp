from setuptools import setup

setup(name='tramp',
      version='0.1',
      author='Antoine Baker',
      author_email='antoine.baker59@gmail.com',
      description='Tree approximate message passing',
      url='https://sphinxteam.github.io/tramp.docs',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[
          'numpy', 'scipy', 'pandas', 'matplotlib', 'daft',
          'networkx>=1.10,<2'
      ])
