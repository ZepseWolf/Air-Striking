from distutils.core import setup
from setuptools import setup, find_packages
 
setup(
    name='example',
    version='0.1',
    author='Author Name', 
    author_email='seekaicong@mail.com',
    package_data={'': ['*.h5','*.pkl']},
    packages=find_packages(exclude=['__pycashe__']),
    include_package_data=True,
    long_description=open('README.md').read()
)