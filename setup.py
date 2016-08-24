import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='pypiw',
      version='0.1',
      description='Testin setuptools, travis and pytest',
      url='http://github.com/hofsmo/PyPiW',
      author='Sigurd Hofsmo Jakobsen',
      author_email='sigurd.jakobsenatgmail.com',
      license='GPLv3',
      packages=['pypiw'],
      zip_safe=False)
