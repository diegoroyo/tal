#!/bin/bash
# Generate updated requirements file
pipreqs . --force
# Generate distribution archives
rm dist/*
python3 setup.py sdist bdist_wheel
# Upload to test pypi
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*