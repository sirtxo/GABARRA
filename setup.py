from setuptools import setup, find_packages

setup(
    name='gabarra',
    version='0.1.0',
    description='Tools for data science',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
)