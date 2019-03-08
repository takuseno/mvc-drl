#!/usr/bin/env python
# coding: utf-8

import uuid

from setuptools import setup, find_packages
from codecs import open
from os import path
try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements

with open('requirements.txt', 'r') as f:
    packages = f.readlines()

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mvc',
    version='1.1.1',
    python_requires='>=3.5',
    install_requires=packages,
    description='Cleanest Deep Reinforcement Learning Implementation Based on Web MVC',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/takuseno/mvc-drl',
    author='takuseno',
    author_email='takuma.seno@gmail.com',
    license='MIT',
    keywords='deep reinforcemtne learning,tensorflow',
    packages=find_packages(exclude=('tests', 'examples')),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
