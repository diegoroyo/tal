#!/bin/bash

# Generate updated requirements file
pipreqs . --force
echo "Please hand-check the requirements.txt file, and modify. Is it OK? (y):"
read a
if [[ $a != 'y' ]]
then
    exit
fi

# Generate distribution archives
rm dist/*
python3 setup.py sdist bdist_wheel
# Upload to test pypi
echo "Uploading to test pypi"
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# Upload to production pypi
echo "Uploading to production pypi"
twine upload dist/*