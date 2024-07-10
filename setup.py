from distutils.core import setup
from setuptools import setup, find_packages
 
setup(
    name='example',
    version='0.1',
    author='Author Name', 
    author_email='seekaicong@mail.com',
    packages=find_packages(),
    long_description=open('README.md').read()
)