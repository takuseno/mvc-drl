#!/bin/bash -eux

# test package
python setup.py test

# build wheel packge
python setup.py bdist_wheel
