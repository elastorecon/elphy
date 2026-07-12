#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.20',
    'scipy>=1.6',
    'h5py>=3.0',
    'scikit-image>=0.18',
    'Click>=6.0',
]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Aaron T Anderson",
    author_email='aandrsn3@illinois.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    description="Material property estimation and analysis for magnetic resonance elastography (MRE) data.",
    entry_points={
        'console_scripts': [
            'elphy=elphy.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='elphy',
    name='elphy',
    packages=find_packages(include=['elphy']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/aaronta/elphy',
    version='0.1.0',
    zip_safe=False,
)
